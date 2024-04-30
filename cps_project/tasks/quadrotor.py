import numpy as np
from isaacgym import gymapi, gymtorch
import torch

from isaacgymenvs.tasks.base.vec_task import VecTask


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
        num_acts = 4

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        super().__init__(
            cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, 13)

        self.root_states = vec_root_tensor
        self.root_positions = vec_root_tensor[..., 0:3]
        self.root_rotations = vec_root_tensor[..., 3:7]
        self.root_linear_velocities = vec_root_tensor[..., 7:10]
        self.root_angular_velocities = vec_root_tensor[..., 10:13]

        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.initial_root_states = vec_root_tensor.clone()

        # TODO: add control tensors

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

    def pre_physics_step(self, _actions):
        pass

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        pass

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        spacing = self.cfg["env"]["envSpacing"]
        envs_per_row = int(np.sqrt(self.num_envs))

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        assets_path = self.cfg["assets"]["path"]
        quad_file = self.cfg["assets"]["quadrotor"]["file"]
        quad_options = gymapi.AssetOptions()
        quad_options.collapse_fixed_joints = self.cfg["assets"]["quadrotor"]["options"][
            "collapse_fixed_joints"
        ]
        quad_options.replace_cylinder_with_capsule = self.cfg["assets"]["quadrotor"][
            "options"
        ]["replace_cylinder_with_capsule"]
        quad_options.flip_visual_attachments = self.cfg["assets"]["quadrotor"][
            "options"
        ]["flip_visual_attachments"]
        quad_options.fix_base_link = self.cfg["assets"]["quadrotor"]["options"][
            "fix_base_link"
        ]
        quad_options.density = self.cfg["assets"]["quadrotor"]["options"]["density"]
        quad_options.angular_damping = self.cfg["assets"]["quadrotor"]["options"][
            "angular_damping"
        ]
        quad_options.linear_damping = self.cfg["assets"]["quadrotor"]["options"][
            "linear_damping"
        ]
        quad_options.max_angular_velocity = self.cfg["assets"]["quadrotor"]["options"][
            "max_angular_velocity"
        ]
        quad_options.max_linear_velocity = self.cfg["assets"]["quadrotor"]["options"][
            "max_linear_velocity"
        ]
        quad_options.disable_gravity = self.cfg["assets"]["quadrotor"]["options"][
            "disable_gravity"
        ]

        quad_asset = self.gym.load_asset(self.sim, assets_path, quad_file, quad_options)
        quad_start_pose = gymapi.Transform()
        quad_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        axis = [-1, 0, 0]
        quad_start_pose.r = gymapi.Quat(
            np.sin(np.pi / 4) * axis[0],
            np.sin(np.pi / 4) * axis[1],
            np.sin(np.pi / 4) * axis[2],
            np.cos(np.pi / 4),
        )

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

        self.envs = []
        self.actors = []
        self.cameras = []
        self.camera_tensors = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, envs_per_row)
            actor = self.gym.create_actor(
                env,
                quad_asset,
                quad_start_pose,
                "quadrotor",
                i,
                self.cfg["assets"]["quadrotor"]["options"]["collision_disabled"],
                0,
            )

            cam = self.gym.create_camera_sensor(env, camera_props)
            self.gym.attach_camera_to_body(
                cam,
                env,
                actor,
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
            self.actors.append(actor)
            self.envs.append(env)
