import numpy as np
import wandb
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import torch

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

from cps_project.controller import Controller
import cv2

import os
from os import getcwd

TARGET_X = 2.5
TARGET_Y = 2.5
TARGET_Z = 2.5


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
        self.camera_type = (
            gymapi.IMAGE_DEPTH
            if self.cfg["sim"]["camera"] == "depth"
            else gymapi.IMAGE_COLOR
        )
        self.image_res = self.image_cfg["resolution"]
        self.flatten_image_sz = self.image_res["width"] * self.image_res["height"]

        # Observations: depth camera image flatten + 13 drone states
        num_obs = self.flatten_image_sz + 13

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
        self.target_dist = torch.zeros(self.num_envs, device=self.device)

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
            [0.0, 0.0, 2.5],
            device=self.device,
        )
        self.root_states[env_ids, 2, 0:3] = torch.tensor(
            [0.0, 5.0, 2.5], device=self.device
        )
        self.root_states[env_ids, 3, 0:3] = torch.tensor(
            [5.0, 2.5, 2.5],
            device=self.device,
        )
        self.root_states[env_ids, 4, 0:3] = torch.tensor(
            [-5.0, 2.5, 2.5], device=self.device
        )

        positions = torch.rand(
            len(env_ids), self.cfg["env"]["numObstacles"], 3, device=self.device
        )
        # print("randomizing positions of obstacles")

        positions[:, :, 0] = positions[:, :, 0] * 10 - 5
        positions[:, :, 1] = positions[:, :, 1] * 5
        positions[:, :, 2] = positions[:, :, 2] * 5

        self.root_states[env_ids, 6:, 0:3] = positions

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
            # print("resetting envs")
            # print(reset_env_ids)
            self.reset_idx(reset_env_ids)

        actions = _actions.to(self.device)
        wandb.log(
            {
                "yaw_rate": actions[0, 3].item(),
                "vel_x": actions[0, 0].item(),
                "vel_y": actions[0, 1].item(),
                "vel_z": actions[0, 2].item(),
            }
        )

        total_torque, common_thrust = self.controller.update(
            actions,
            self.root_rotations[:, 0],
            self.root_angular_velocities[:, 0],
            self.root_linear_velocities[:, 0],
        )

        friction = (
            -0.5
            * torch.sign(self.root_linear_velocities[:, 0])
            * self.root_linear_velocities[:, 0] ** 2
        )

        self.forces[:, 0] += friction
        self.forces[:, 0, 2] = common_thrust
        self.torques[:, 0] = total_torque

        # No controller
        # self.forces[:, 0] += actions[:, 0:3]
        # self.torques[:, 0, 2] = actions[:, 3]

        wandb.log(
            {
                "thrust": common_thrust[0].item(),
                "torque_x": total_torque[0, 0].item(),
                "torque_y": total_torque[0, 1].item(),
                "torque_z": total_torque[0, 2].item(),
            }
        )

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

        small_sphere_asset = self.gym.load_asset(
            self.sim,
            self.assets_path,
            "obstacles/objects/small_sphere.urdf",
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
        sphere_color = gymapi.Vec3(200 / 255, 210 / 255, 100 / 255)

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
                0,  # collision group
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

            self.cameras.append(cam)

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

            sphere_pose = gymapi.Transform()
            sphere_pose.p = gymapi.Vec3(TARGET_X, TARGET_Y, TARGET_Z)
            small_sphere = self.gym.create_actor(
                env,
                small_sphere_asset,
                sphere_pose,
                "small_sphere",
                i,
                0,
                0,
            )

            self.gym.set_rigid_body_color(
                env,
                small_sphere,
                0,
                gymapi.MESH_VISUAL,
                sphere_color,
            )

            self.actors.append(small_sphere)
            self.actors.append(left_wall)
            self.actors.append(right_wall)
            self.actors.append(back_wall)
            self.actors.append(front_wall)

            for j in range(self.cfg["env"]["numObstacles"]):
                cube = self.gym.create_actor(
                    env,
                    cube_asset,
                    quad_start_pose,
                    f"cube{j}",
                    i,
                    0,
                    3,
                )
                color = np.random.randint(low=50, high=200, size=3)

                self.gym.set_rigid_body_color(
                    env,
                    cube,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(color[0] / 255, color[1] / 255, color[2] / 255),
                )

                self.actors.append(cube)

            self.envs.append(env)

    def compute_observations(self):
        for env_id in range(self.num_envs):
            self.obs_buf[env_id, 0 : self.flatten_image_sz] = self.full_camera_array[
                env_id
            ].view(-1)
            self.obs_buf[env_id, self.flatten_image_sz :] = self.root_states[env_id, 0]
            # img = self.obs_buf[env_id].detach().cpu().numpy()
            # img = img.reshape(self.image_res["height"], self.image_res["width"])
            # img = img * 255
            # img = img.astype(np.uint8)
            # img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            # # save image with cv2
            # path = os.path.join(getcwd(), "img", f"{env_id}.png")
            # print(f"saving image to {path}")
            # cv2.imwrite(path, img)

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
            self.target_dist,
        )


#####################################################################
###=========================jit functions=========================###
#####################################################################
# @torch.jit.script
def compute_quadcopter_reward(
    root_positions,
    root_quats,
    root_linvels,
    root_angvels,
    reset_buf,
    progress_buf,
    max_episode_length,
    old_target_dist,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor) -> Tuple[Tensor, Tensor]

    target_dist = torch.sqrt(
        (TARGET_X - root_positions[..., 0]) * (TARGET_X - root_positions[..., 0])
        + (TARGET_Y - root_positions[..., 1]) * (TARGET_Y - root_positions[..., 1])
        + (TARGET_Z - root_positions[..., 2]) * (TARGET_Z - root_positions[..., 2])
    )
    pos_reward = 1.0 / (1.0 + target_dist * target_dist)

    # uprightness
    ups = quat_axis(root_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 1.0 / (1.0 + tiltage * tiltage)

    # spinning
    spinnage = torch.norm(root_angvels)
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    vel = torch.norm(root_linvels, dim=-1)
    vel_reward = 1.0 / (1.0 + vel * vel)

    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = pos_reward + pos_reward * (up_reward + spinnage_reward + vel_reward)
    # reward = pos_reward

    wandb.log(
        {
            "reward": reward.mean().item(),
            "pos_reward": pos_reward.mean().item(),
            "up_reward": up_reward.mean().item(),
            "spinnage_reward": spinnage_reward.mean().item(),
            "vel_reward": vel_reward.mean().item(),
            "target_dist": target_dist[0].item(),
        }
    )

    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    # die = torch.where(target_dist > 3.0, ones, die)

    die = torch.where(root_positions[:, 2] < 0.1, ones, die)
    # die = torch.where(
    #     target_dist > 3.5,
    #     ones,
    #     die,
    # )

    die = torch.where(
        root_positions[..., 2] > 5,
        ones,
        die,
    )

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length // 2 - 1, ones, die)

    return reward, reset
