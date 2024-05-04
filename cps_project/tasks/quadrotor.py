import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import torch

from isaacgymenvs.tasks.base.vec_task import VecTask

from cps_project.utils.controller import Controller


class Quadrotor(VecTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.image_cfg = self.cfg["env"]["image"]
        self.assets_path = self.cfg["assets"]["path"]

        # Observations: depth camera image
        num_obs = (
            self.image_cfg["resolution"]["width"]
            * self.image_cfg["resolution"]["height"]
        )

        # Actions:
        # 0. Roll
        # 1. Pitch
        # 2. Yaw
        # 3. Common thrust
        num_actions = 4

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_actions

        super().__init__(
            cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        self.actors_per_env = len(self.actors) // self.num_envs

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(
            self.num_envs, self.actors_per_env, 13
        )

        self.root_states = vec_root_tensor
        self.root_positions = vec_root_tensor[..., 0:3]
        self.root_rotations = vec_root_tensor[..., 3:7]
        self.root_linear_velocities = vec_root_tensor[..., 7:10]
        self.root_angular_velocities = vec_root_tensor[..., 10:13]

        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.initial_root_states = vec_root_tensor.clone()

        max_thrust = 2
        self.thrust_lower_limits = torch.zeros(
            4, device=self.device, dtype=torch.float32
        )
        self.thrust_upper_limits = max_thrust * torch.ones(
            4, device=self.device, dtype=torch.float32
        )

        self.forces = torch.zeros(
            (self.num_envs, self.actors_per_env, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.torques = torch.zeros(
            (self.num_envs, self.actors_per_env, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        self.controller = Controller(self.num_envs, self.device)

        self.all_actor_indices = torch.arange(
            self.num_envs, dtype=torch.int32, device=self.device
        )

        if self.viewer:
            cam_pos = gymapi.Vec3(1.0, 1.0, 1.8)
            cam_target = gymapi.Vec3(2.2, 2.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_sim(self):
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self.dt = self.sim_params.dt
        self._create_ground_plane()
        self._create_envs()

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        actor_indices = self.all_actor_indices[env_ids].flatten()

        self.root_states[env_ids] = self.initial_root_states[env_ids]

        print(
            "self.root_states[env_ids] before rand_float: ", self.root_states[env_ids]
        )

        self.root_states[env_ids, 0, 0:2] = torch_rand_float(
            0.0, 0.0, (num_resets, 2), self.device
        )
        self.root_states[env_ids, 0, 2] = 1.0

        # self.root_states[env_ids, 0, 3:6] = torch_rand_float(
        #     0.0, 0.0, (num_resets, 3), self.device
        # )
        # self.root_states[env_ids, 0, 6] = 1.0

        self.root_states[env_ids, 0, 7:10] = torch_rand_float(
            0.0, 0.0, (num_resets, 3), self.device
        )
        self.root_states[env_ids, 0, 10:13] = torch_rand_float(
            0.0, 0.0, (num_resets, 3), self.device
        )

        print("self.root_states[env_ids] after rand_float: ", self.root_states[env_ids])

        # quad_start_pose = 0.001 * torch.zeros((num_resets, 4), device=self.device)
        # # quad_start_pose[:, 0] = -np.sin(np.pi / 4)
        # # quad_start_pose[:, 3] = np.cos(np.pi / 4)
        # quad_start_pose[:, -1] = 1.0

        # self.root_states[env_ids, 0, 3:7] = quad_start_pose

        print("self.root_states[env_ids] after start pose: ", self.root_states[env_ids])

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            self.root_tensor,
            gymtorch.unwrap_tensor(actor_indices),
            num_resets,
        )

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, _actions):
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            print("len(reset_env_ids): ", len(reset_env_ids))
            print("reset_env_ids: ", reset_env_ids)
            self.reset_idx(reset_env_ids)

        # actions = _actions.to(self.device)

        # print("\nself.root_states[:, 0]: ", self.root_states[:, 0], "\n")

        # total_torque, common_thrust = self.controller.update(
        #     actions,
        #     self.root_rotations[:, 0],
        #     self.root_angular_velocities[:, 0],
        #     self.root_linear_velocities[:, 0],
        # )

        # friction = (
        #     -0.02
        #     * torch.sign(self.root_linear_velocities[:, 0])
        #     * self.root_linear_velocities[:, 0] ** 2
        # )

        # self.forces[:, 0] = friction
        # # print("\nCOMMON THRUST: ", common_thrust, "\n")
        # self.forces[:, 0, 2] += common_thrust
        # self.torques[:, 0] = total_torque

        # # clear actions for reset envs
        # self.forces[reset_env_ids] = 0.0
        # self.torques[reset_env_ids] = 0.0

        print("self.forces: ", self.forces)
        print("self.torques: ", self.torques)

        # apply actions
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.forces),
            gymtorch.unwrap_tensor(self.torques),
            gymapi.LOCAL_SPACE,
        )

    def post_physics_step(self):
        self.progress_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        spacing = self.cfg["env"]["envSpacing"]
        envs_per_row = int(np.sqrt(self.num_envs))

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        quad_cfg = self.cfg["assets"]["quadrotor"]
        quad_file = quad_cfg["file"]
        quad_cfg_options = quad_cfg["options"]

        quad_options = gymapi.AssetOptions()
        quad_options.fix_base_link = quad_cfg_options["fix_base_link"]
        quad_options.angular_damping = quad_cfg_options["angular_damping"]
        quad_options.max_angular_velocity = quad_cfg_options["max_angular_velocity"]
        quad_options.disable_gravity = quad_cfg_options["disable_gravity"]

        quad_asset = self.gym.load_asset(
            self.sim, self.assets_path, quad_file, quad_options
        )
        quad_start_pose = gymapi.Transform()
        quad_start_pose.p.z = 1.0

        # Set Camera Properties
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = self.image_cfg["resolution"]["width"]
        camera_props.height = self.image_cfg["resolution"]["height"]
        camera_props.far_plane = 15.0
        camera_props.horizontal_fov = 87.0
        # local camera transform
        local_transform = gymapi.Transform()
        # position of the camera relative to the body
        local_transform.p = gymapi.Vec3(0.15, 0.00, 0.05)
        # orientation of the camera relative to the body
        local_transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        wall_options = gymapi.AssetOptions()
        wall_options.fix_base_link = True
        wall_asset = self.gym.load_asset(
            self.sim,
            self.assets_path,
            "obstacles/walls/left_wall.urdf",
            wall_options,
        )

        self.envs = []
        self.actors = []
        self.cameras = []
        self.camera_tensors = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, envs_per_row)
            quad = self.gym.create_actor(
                env,
                quad_asset,
                quad_start_pose,
                "quadrotor",
                i,
                quad_cfg_options["collision_disabled"],
                0,
            )
            self.actors.append(quad)

            cam = self.gym.create_camera_sensor(env, camera_props)
            self.gym.attach_camera_to_body(
                cam,
                env,
                quad,
                local_transform,
                gymapi.FOLLOW_TRANSFORM,
            )
            camera_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim,
                env,
                cam,
                # IMAGE_COLOR -> RGBA, IMAGE_DEPTH -> Depth
                gymapi.IMAGE_DEPTH
                if self.cfg["sim"]["camera"] == "depth"
                else gymapi.IMAGE_COLOR,
            )
            torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
            self.camera_tensors.append(torch_cam_tensor)

            # wall_pose = gymapi.Transform()
            # wall_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            # wall = self.gym.create_actor(
            #     env,
            #     wall_asset,
            #     wall_pose,
            #     "wall",
            #     i,
            #     False,
            #     1,
            # )
            # self.actors.append(wall)

            self.cameras.append(cam)
            self.envs.append(env)
