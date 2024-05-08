import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import torch

from isaacgymenvs.utils.torch_jit_utils import *
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
        # num_obs = (
        #     self.image_cfg["resolution"]["width"]
        #     * self.image_cfg["resolution"]["height"]
        # )
        num_obs = 13

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
        self.contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(
            self.num_envs, self.actors_per_env, 13
        )

        self.root_states = vec_root_tensor
        self.root_positions = vec_root_tensor[..., 0:3]
        self.root_rotations = vec_root_tensor[..., 3:7]
        self.root_linear_velocities = vec_root_tensor[..., 7:10]
        self.root_angular_velocities = vec_root_tensor[..., 10:13]

        self.contact_forces = gymtorch.wrap_tensor(self.contact_force_tensor).view(
            self.num_envs, self.actors_per_env, 3
        )[:, 0]

        self.collisions = torch.zeros(self.num_envs, device=self.device)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

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

        self.all_actor_indices = torch.tensor(
            (self.num_envs, self.actors_per_env),
            dtype=torch.int32,
            device=self.device,
        )

        if self.viewer:
            cam_pos = gymapi.Vec3(1.0, 1.0, 20.0)
            cam_target = gymapi.Vec3(0.0, 1.0, 1.0)
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
        self.root_states[env_ids] = self.initial_root_states[env_ids]

        # z needs to be 1.0 otherwise the quadrotor will fly away like crazy
        self.root_states[env_ids, 0, 0:3] = torch.tensor(
            [1.0, 1.0, 1.0], device=self.device
        )
        self.root_states[env_ids, 0, 3:7] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=self.device
        )
        self.root_states[env_ids, 0, 7:10] = 0.0
        self.root_states[env_ids, 0, 10:13] = 0.0

        self.root_states[env_ids, 1, 0:3] = torch.tensor(
            [5.0, 0.0, 2.5],
            device=self.device,
        )
        self.root_states[env_ids, 2, 0:3] = torch.tensor(
            [5.0, 5.0, 2.5], device=self.device
        )
        self.root_states[env_ids, 3, 0:3] = torch.tensor(
            [10.0, 2.5, 2.5],
            device=self.device,
        )
        self.root_states[env_ids, 4, 0:3] = torch.tensor(
            [0.0, 2.5, 2.5], device=self.device
        )

        positions = torch.rand(
            self.num_envs, self.cfg["env"]["numObstacles"], 3, device=self.device
        )
        positions[:, :, 0] = positions[:, :, 0] * 10
        positions[:, :, 1] = positions[:, :, 1] * 5
        positions[:, :, 2] = positions[:, :, 2] * 5

        # self.root_states[:, 5:, 0:3] = positions

        self.gym.set_actor_root_state_tensor(
            self.sim,
            self.root_tensor,
        )

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def check_collisions(self):
        ones = torch.ones((self.num_envs), device=self.device)
        zeros = torch.zeros((self.num_envs), device=self.device)
        self.collisions[:] = 0
        self.collisions = torch.where(
            torch.norm(self.contact_forces, dim=1) > 0.1, ones, zeros
        )

        ones = torch.ones_like(self.reset_buf)
        self.reset_buf = torch.where(self.collisions > 0, ones, self.reset_buf)

    def pre_physics_step(self, _actions):
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            print("\nRESET ENV IDS: ", reset_env_ids, "\n")
            self.reset_idx(reset_env_ids)

        actions = _actions.to(self.device)

        total_torque, common_thrust = self.controller.update(
            actions,
            self.root_rotations[:, 0],
            self.root_angular_velocities[:, 0],
            self.root_linear_velocities[:, 0],
        )

        friction = (
            -0.02
            * torch.sign(self.root_linear_velocities[:, 0])
            * self.root_linear_velocities[:, 0] ** 2
        )

        self.forces[:, 0] = friction
        # print("\nCOMMON THRUST: ", common_thrust, "\n")
        self.forces[:, 0, 2] += common_thrust
        self.torques[:, 0] = total_torque

        # clear actions for reset envs
        self.forces[reset_env_ids] = 0.0
        self.torques[reset_env_ids] = 0.0

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
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.check_collisions()
        self.compute_observations()
        self.compute_reward()

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        spacing = self.cfg["env"]["envSpacing"]
        envs_per_row = int(np.sqrt(self.num_envs))

        self.env_lower_bound = gymapi.Vec3(-spacing, -spacing, 0.0)
        self.env_upper_bound = gymapi.Vec3(spacing, spacing, spacing)

        quad_cfg = self.cfg["assets"]["quadrotor"]
        quad_file = quad_cfg["file"]

        quad_asset = self.gym.load_asset(
            self.sim, self.assets_path, quad_file, gymapi.AssetOptions()
        )
        quad_start_pose = gymapi.Transform()
        quad_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)

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

        obstacle_options = gymapi.AssetOptions()
        obstacle_options.fix_base_link = True

        left_wall_asset = self.gym.load_asset(
            self.sim,
            self.assets_path,
            "obstacles/walls/left_wall.urdf",
            obstacle_options,
        )
        right_wall_asset = self.gym.load_asset(
            self.sim,
            self.assets_path,
            "obstacles/walls/right_wall.urdf",
            obstacle_options,
        )
        front_wall_asset = self.gym.load_asset(
            self.sim,
            self.assets_path,
            "obstacles/walls/front_wall.urdf",
            obstacle_options,
        )
        back_wall_asset = self.gym.load_asset(
            self.sim,
            self.assets_path,
            "obstacles/walls/back_wall.urdf",
            obstacle_options,
        )

        cube_asset = self.gym.load_asset(
            self.sim,
            self.assets_path,
            "obstacles/objects/small_cube.urdf",
            obstacle_options,
        )

        self.envs = []
        self.actors = []
        self.cameras = []
        self.camera_tensors = []

        wall_color = gymapi.Vec3(100 / 255, 200 / 255, 210 / 255)

        for i in range(self.num_envs):
            env = self.gym.create_env(
                self.sim, self.env_lower_bound, self.env_upper_bound, envs_per_row
            )
            quad = self.gym.create_actor(
                env,
                quad_asset,
                quad_start_pose,
                "quadrotor",
                i,
                1,
                0,
            )
            self.actors.append(quad)

            left_wall = self.gym.create_actor(
                env,
                left_wall_asset,
                quad_start_pose,
                "left_wall",
                i,
                0,
                8,
            )
            self.gym.set_rigid_body_color(
                env,
                left_wall,
                0,
                gymapi.MESH_VISUAL,
                wall_color,
            )

            right_wall = self.gym.create_actor(
                env,
                right_wall_asset,
                quad_start_pose,
                "right_wall",
                i,
                0,
                8,
            )
            self.gym.set_rigid_body_color(
                env,
                right_wall,
                0,
                gymapi.MESH_VISUAL,
                wall_color,
            )

            back_wall = self.gym.create_actor(
                env,
                back_wall_asset,
                quad_start_pose,
                "back_wall",
                i,
                0,
                8,
            )
            self.gym.set_rigid_body_color(
                env,
                back_wall,
                0,
                gymapi.MESH_VISUAL,
                wall_color,
            )

            front_wall = self.gym.create_actor(
                env,
                front_wall_asset,
                quad_start_pose,
                "front_wall",
                i,
                0,
                8,
            )
            self.gym.set_rigid_body_color(
                env,
                front_wall,
                0,
                gymapi.MESH_VISUAL,
                wall_color,
            )

            self.actors.append(left_wall)
            self.actors.append(right_wall)
            self.actors.append(back_wall)
            self.actors.append(front_wall)

            # for j in range(self.cfg["env"]["numObstacles"]):
            #     cube = self.gym.create_actor(
            #         env,
            #         cube_asset,
            #         quad_start_pose,
            #         f"cube{j}",
            #         i,
            #         0,
            #         3,
            #     )
            #     color = np.random.randint(low=50, high=200, size=3)

            #     self.gym.set_rigid_body_color(
            #         env,
            #         cube,
            #         0,
            #         gymapi.MESH_VISUAL,
            #         gymapi.Vec3(color[0] / 255, color[1] / 255, color[2] / 255),
            #     )

            #     self.actors.append(cube)

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

            self.cameras.append(cam)
            self.envs.append(env)

    def compute_observations(self):
        target_x = 0.0
        target_y = 0.0
        target_z = 5.0
        self.obs_buf[..., 0] = (target_x - self.root_positions[..., 0, 0]) / 3
        self.obs_buf[..., 1] = (target_y - self.root_positions[..., 0, 1]) / 3
        self.obs_buf[..., 2] = (target_z - self.root_positions[..., 0, 2]) / 3
        self.obs_buf[..., 3:7] = self.root_rotations[..., 0, :]
        self.obs_buf[..., 7:10] = self.root_linear_velocities[..., 0, :] / 2
        self.obs_buf[..., 10:13] = self.root_angular_velocities[..., 0, :] / torch.pi
        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_quadcopter_reward(
            self.root_positions[..., 0, :],
            self.root_rotations[..., 0, :],
            self.root_linear_velocities[..., 0, :],
            self.root_angular_velocities[..., 0, :],
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length,
        )
        print("reset_buf: ", self.reset_buf)


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_quadcopter_reward(
    root_positions,
    root_quats,
    root_linvels,
    root_angvels,
    reset_buf,
    progress_buf,
    max_episode_length,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # distance to target
    target_dist = torch.sqrt(
        root_positions[..., 0] * root_positions[..., 0]
        + root_positions[..., 1] * root_positions[..., 1]
        + (1 - root_positions[..., 2]) * (1 - root_positions[..., 2])
    )
    pos_reward = 1.0 / (1.0 + target_dist * target_dist)

    # uprightness
    ups = quat_axis(root_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 1.0 / (1.0 + tiltage * tiltage)

    # spinning
    spinnage = torch.abs(root_angvels[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = pos_reward + pos_reward * (up_reward + spinnage_reward)

    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    # die = torch.where(target_dist > 3.0, ones, die)
    die = torch.where(root_positions[..., 2] < 0.1, ones, die)
    die = torch.where(root_positions[..., 2] > 10.0, ones, die)

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward, reset
