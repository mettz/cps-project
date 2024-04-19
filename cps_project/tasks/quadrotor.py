import torch
from isaacgym import gymapi, gymtorch

from .base.vec_task import VecTask


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
        num_obs = self.image_cfg.resolution.width * self.image_cfg.resolution.height

        # Actions:
        # 0. Roll
        # 1. Pitch
        # 2. Yaw
        # 3. Common thrust
        num_acts = 4

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        self.root_tensor = self.gym.acquire_actior_root_state_tensor(self.sim)
        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, 13)

        self.root_states = vec_root_tensor
        self.root_positions = vec_root_tensor[..., 0:3]
        self.root_rotations = vec_root_tensor[..., 3:7]
        self.root_linear_velocities = vec_root_tensor[..., 7:10]
        self.root_angular_velocities = vec_root_tensor[..., 10:13]

        self.gym.refresh_actor_root_state_tensor(self.sim, self.root_tensor)

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

    def pre_physics_step(self):
        pass

    def post_physics_step(self):
        pass

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        spacing = self.cfg["env"]["envSpacing"]

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_path = self.cfg["assets"]["path"]

        self.envs = []
        self.env_actor_handles = []
        self.env_actor_states = []
        self.env_actor_controls = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, self.cfg["env"]["envParams"])
            self.envs.append(env)

            actor_handle = self.gym.create_actor(env, self.cfg["env"]["actorParams"])
            self.env_actor_handles.append(actor_handle)

            actor_state = self.gym.acquire_actor_state_tensor(env, actor_handle)
            actor_state = gymtorch.wrap_tensor(actor_state).view(1, -1)
            self.env_actor_states.append(actor_state)

            actor_controls = self.gym.acquire_actor_controls_tensor(env, actor_handle)
            actor_controls = gymtorch.wrap_tensor(actor_controls).view(1, -1)
            self.env_actor_controls.append(actor_controls)

            self.gym.set_actor_root_state_tensor(env, actor_handle, self.root_tensor)
