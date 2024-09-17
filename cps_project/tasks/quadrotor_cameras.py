import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import torch

from isaacgymenvs.utils.torch_jit_utils import quat_axis

from cps_project.tasks.quadrotor import Quadrotor

# The `Quadrotor` class implements an IsaacGym task with an environment in which a quadrotor
# tries to avoid obstacles and reach a target using cameras. The task can be used for
# training reinforcement learning agents e.g. using the `SKRL` RL library.

TARGET_X = 2.5  # target X position in meters
TARGET_Y = 2.5  # target Y position in meters
TARGET_Z = 2.5  # target Z position in meters


class QuadrotorCameras(Quadrotor):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
        wandb=False,
    ):
        self.image_cfg = cfg["env"]["image"]
        self.image_res = self.image_cfg["resolution"]
        self.flatten_image_sz = self.image_res["width"] * self.image_res["height"]

        if cfg["sim"]["camera"] != "depth":
            self.flatten_image_sz *= 4

        # Observations: depth camera image flatten + 13 drone states
        num_obs = self.flatten_image_sz + 13
        super().__init__(
            cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
            num_obs=num_obs,
            wandb=wandb,
        )

        self.compute_quadcopter_reward_fn = compute_quadcopter_cameras_reward

        self.camera_type = (
            gymapi.IMAGE_DEPTH
            if self.cfg["sim"]["camera"] == "depth"
            else gymapi.IMAGE_COLOR
        )

        if self.camera_type == gymapi.IMAGE_DEPTH:
            self.full_camera_array = torch.zeros(
                self.num_envs,
                self.image_res["height"],
                self.image_res["width"],
                device=self.device,
            )
        else:
            self.full_camera_array = torch.zeros(
                self.num_envs,
                self.image_res["height"],
                self.image_res["width"],
                4,
                device=self.device,
            )

    # This method is called when there are environments that need to be reset to their initial state.
    # In particular, it resets the quadrotor to its initial position and orientation and its linear
    # and angular velocities to zero.
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

    def dump_images(self):
        for env_id in range(self.num_envs):
            # the depth values are in -ve z axis, so we need to flip it to positive
            if self.camera_type == gymapi.IMAGE_DEPTH:
                self.full_camera_array[env_id] = -self.camera_tensors[env_id]
            else:
                self.full_camera_array[env_id] = self.camera_tensors[env_id]

    def render_cameras(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.dump_images()
        self.gym.end_access_image_tensors(self.sim)
        return

    def post_physics_step(self):
        self.render_cameras()
        super().post_physics_step()

    def _load_assets(self):
        super()._load_assets()

        cube_options = gymapi.AssetOptions()
        cube_options.fix_base_link = True
        self.cube_asset = self.gym.load_asset(
            self.sim,
            self.assets_path,
            "obstacles/objects/small_cube.urdf",
            cube_options,
        )

    def _create_env(self, env_id):
        super()._create_env(env_id)

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

        quad = self.actors[env_id][0]
        cam = self.gym.create_camera_sensor(self.envs[env_id], camera_props)
        self.gym.attach_camera_to_body(
            cam,
            self.envs[env_id],
            quad,
            local_transform,
            gymapi.FOLLOW_TRANSFORM,
        )

        camera_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim,
            self.envs[env_id],
            cam,
            # IMAGE_COLOR -> RGBA, IMAGE_DEPTH -> Depth
            gymapi.IMAGE_DEPTH
            if self.cfg["sim"]["camera"] == "depth"
            else gymapi.IMAGE_COLOR,
        )

        torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
        self.camera_tensors.append(torch_cam_tensor)

        self.cameras.append(cam)

        obs_pos = torch.rand(
            len(self.envs), self.cfg["env"]["numObstacles"], 3, device=self.device
        )
        obs_pos[:, :, 0] = obs_pos[:, :, 0] * 7 - 3
        obs_pos[:, :, 1] = obs_pos[:, :, 1] * 3 + 1
        obs_pos[:, :, 2] = obs_pos[:, :, 2] * 3 + 1

        for j in range(self.cfg["env"]["numObstacles"]):
            cube = self.gym.create_actor(
                self.envs[env_id],
                self.cube_asset,
                gymapi.Transform(
                    p=gymapi.Vec3(
                        obs_pos[env_id, j, 0],
                        obs_pos[env_id, j, 1],
                        obs_pos[env_id, j, 2],
                    )
                ),
                f"cube{j}",
                env_id,
                0,
                3,
            )
            color = np.random.randint(low=50, high=200, size=3)

            self.gym.set_rigid_body_color(
                self.envs[env_id],
                cube,
                0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(color[0] / 255, color[1] / 255, color[2] / 255),
            )

            self.actors[env_id].append(cube)

    def _create_envs(self):
        self.cameras = []
        self.camera_tensors = []

        super()._create_envs()

    def compute_observations(self):
        for env_id in range(self.num_envs):
            self.obs_buf[env_id, 0 : self.flatten_image_sz] = self.full_camera_array[
                env_id
            ].view(-1)
            self.obs_buf[env_id, self.flatten_image_sz :] = self.root_states[env_id, 0]

        return self.obs_buf


@torch.jit.script
def compute_quadcopter_cameras_reward(
    target,
    root_positions,
    root_quats,
    root_linvels,
    root_angvels,
    reset_buf,
    progress_buf,
    max_episode_length,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    target_dist = torch.sqrt(
        (target[0] - root_positions[..., 0]) * (target[0] - root_positions[..., 0])
        + (target[1] - root_positions[..., 1]) * (target[1] - root_positions[..., 1])
        + (target[2] - root_positions[..., 2]) * (target[2] - root_positions[..., 2])
    )
    pos_reward = 1.0 / (1.0 + target_dist * target_dist)

    # uprightness
    ups = quat_axis(root_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 1.0 / (1.0 + tiltage * tiltage)

    # spinning
    spinnage = torch.abs(root_angvels[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    vel = torch.norm(root_linvels, dim=-1)
    vel_reward = 1.0 / (1.0 + vel * vel)

    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = pos_reward + pos_reward * (up_reward + spinnage_reward + vel_reward)

    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)

    die = torch.where(root_positions[:, 2] < 0.1, ones, die)
    die = torch.where(
        root_positions[..., 2] > 5,
        ones,
        die,
    )

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return (
        reward,
        pos_reward,
        up_reward,
        spinnage_reward,
        vel_reward,
        target_dist,
        reset,
    )
