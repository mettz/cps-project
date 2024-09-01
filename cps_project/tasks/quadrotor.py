import numpy as np
import wandb
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import torch

from isaacgymenvs.utils.torch_jit_utils import quat_axis
from isaacgymenvs.tasks.base.vec_task import VecTask

from cps_project.controller import Controller

# The `Quadrotor` class implements an IsaacGym task with an environment in which
# a quadrotor tries to reach a target without using cameras. The task can be used
# for training reinforcement learning agents e.g. using the `SKRL` RL library.

TARGET_X = 2.5  # target X position in meters
TARGET_Y = 2.5  # target Y position in meters
TARGET_Z = 2.5  # target Z position in meters


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
        wandb=False,
        num_obs=None,
    ):
        self.cfg = cfg
        self.wandb = wandb

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.assets_path = self.cfg["assets"]["path"]

        # Observations correspondds to the quadrotor's state vector:
        # 1. quadrotor position in the world frame (x, y, z)
        # 2. quadrotor orientation as a quaternion (x, y, z, w)
        # 3. quadrotor linear velocity (Vx, Vy, Vz)
        # 4. quadrotor angular velocity (ωx, ωy, ωz)
        self.cfg["env"]["numObservations"] = 13 if num_obs is None else num_obs

        # Actions:
        # 0. Roll
        # 1. Pitch
        # 2. Yaw
        # 3. Common thrust
        self.cfg["env"]["numActions"] = 4

        super().__init__(
            cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # self.actors is populated with the actors of the environment when the self.create_sim()
        # method is called which happens in the super().__init__() call
        self.actors_per_env = len(self.actors[0])
        print(f"actors per env: {self.actors_per_env}")

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

        # we only care about the contact forces of the quadrotor (actor 0)
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

        self.target_tensor = torch.tensor(
            [TARGET_X, TARGET_Y, TARGET_Z], device=self.device
        )
        self.compute_quadcopter_reward_fn = compute_quadcopter_reward

        self.controller = Controller(self.num_envs, self.device)

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
        self._create_ground_plane()
        self._create_envs()

    # This method is called when there are environments that need to be reset to their initial state.
    # In particular, it resets the quadrotor to its initial position and orientation and its linear
    # and angular velocities to zero.
    def reset_idx(self, env_ids):
        self.root_states[env_ids] = self.initial_root_states[env_ids]

        # z needs to be 1.0 otherwise the quadrotor will fly away immediately without control
        self.root_states[env_ids, 0, 0:3] = torch.tensor(
            [1.0, 1.0, 1.0], device=self.device
        )
        self.root_states[env_ids, 0, 3:7] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=self.device
        )
        self.root_states[env_ids, 0, 7:10] = 0.0
        self.root_states[env_ids, 0, 10:13] = 0.0

    def check_collisions(self):
        ones = torch.ones((self.num_envs), device=self.device)
        zeros = torch.zeros((self.num_envs), device=self.device)
        self.collisions[:] = 0

        # we consider a collision if the contact force is greater than 0.1 N
        self.collisions = torch.where(
            torch.norm(self.contact_forces, dim=1) > 0.1, ones, zeros
        )

        ones = torch.ones_like(self.reset_buf)

        # if a collision is detected, reset the environment
        self.reset_buf = torch.where(self.collisions > 0, ones, self.reset_buf)

    def pre_physics_step(self, _actions):
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
            self.gym.set_actor_root_state_tensor(
                self.sim,
                self.root_tensor,
            )

            # clear progress and reset buffers to prepare for
            # the next episode
            self.reset_buf[reset_env_ids] = 0
            self.progress_buf[reset_env_ids] = 0

        actions = _actions.to(self.device)
        if self.wandb:
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
        if self.wandb:
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

    def _load_assets(self):
        quad_cfg = self.cfg["assets"]["quadrotor"]
        quad_file = quad_cfg["file"]

        self.quad_asset = self.gym.load_asset(
            self.sim, self.assets_path, quad_file, gymapi.AssetOptions()
        )

        wall_options = gymapi.AssetOptions()
        wall_options.fix_base_link = True

        self.left_wall_asset = self.gym.load_asset(
            self.sim,
            self.assets_path,
            "obstacles/walls/left_wall.urdf",
            wall_options,
        )
        self.right_wall_asset = self.gym.load_asset(
            self.sim,
            self.assets_path,
            "obstacles/walls/right_wall.urdf",
            wall_options,
        )
        self.front_wall_asset = self.gym.load_asset(
            self.sim,
            self.assets_path,
            "obstacles/walls/front_wall.urdf",
            wall_options,
        )
        self.back_wall_asset = self.gym.load_asset(
            self.sim,
            self.assets_path,
            "obstacles/walls/back_wall.urdf",
            wall_options,
        )
        self.small_sphere_asset = self.gym.load_asset(
            self.sim,
            self.assets_path,
            "obstacles/objects/small_sphere.urdf",
            wall_options,
        )

    def _create_env(self, env_id):
        quad_start_pose = gymapi.Transform()
        quad_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        quad = self.gym.create_actor(
            self.envs[env_id],
            self.quad_asset,
            quad_start_pose,
            "quadrotor",
            env_id,
            1,
            0,  # collision group
        )
        self.actors[env_id].append(quad)

        wall_color = gymapi.Vec3(100 / 255, 200 / 255, 210 / 255)

        left_wall = self.gym.create_actor(
            self.envs[env_id],
            self.left_wall_asset,
            gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 2.5)),
            "left_wall",
            env_id,
            0,
            8,
        )
        self.gym.set_rigid_body_color(
            self.envs[env_id],
            left_wall,
            0,
            gymapi.MESH_VISUAL,
            wall_color,
        )

        right_wall = self.gym.create_actor(
            self.envs[env_id],
            self.right_wall_asset,
            gymapi.Transform(p=gymapi.Vec3(0.0, 5.0, 2.5)),
            "right_wall",
            env_id,
            0,
            8,
        )
        self.gym.set_rigid_body_color(
            self.envs[env_id],
            right_wall,
            0,
            gymapi.MESH_VISUAL,
            wall_color,
        )

        back_wall = self.gym.create_actor(
            self.envs[env_id],
            self.back_wall_asset,
            gymapi.Transform(p=gymapi.Vec3(5.0, 2.5, 2.5)),
            "back_wall",
            env_id,
            0,
            8,
        )
        self.gym.set_rigid_body_color(
            self.envs[env_id],
            back_wall,
            0,
            gymapi.MESH_VISUAL,
            wall_color,
        )

        front_wall = self.gym.create_actor(
            self.envs[env_id],
            self.front_wall_asset,
            gymapi.Transform(p=gymapi.Vec3(-5.0, 2.5, 2.5)),
            "front_wall",
            env_id,
            0,
            8,
        )

        self.gym.set_rigid_body_color(
            self.envs[env_id],
            front_wall,
            0,
            gymapi.MESH_VISUAL,
            wall_color,
        )

        sphere_pose = gymapi.Transform()
        sphere_pose.p = gymapi.Vec3(TARGET_X, TARGET_Y, TARGET_Z)
        small_sphere = self.gym.create_actor(
            self.envs[env_id],
            self.small_sphere_asset,
            sphere_pose,
            "small_sphere",
            env_id,
            0,
            0,
        )

        sphere_color = gymapi.Vec3(200 / 255, 210 / 255, 100 / 255)
        self.gym.set_rigid_body_color(
            self.envs[env_id],
            small_sphere,
            0,
            gymapi.MESH_VISUAL,
            sphere_color,
        )

        self.actors[env_id].append(small_sphere)
        self.actors[env_id].append(left_wall)
        self.actors[env_id].append(right_wall)
        self.actors[env_id].append(back_wall)
        self.actors[env_id].append(front_wall)

    def _create_envs(self):
        spacing = self.cfg["env"]["envSpacing"]
        envs_per_row = int(np.sqrt(self.num_envs))

        self.env_lower_bound = gymapi.Vec3(-spacing, -spacing, 0.0)
        self.env_upper_bound = gymapi.Vec3(spacing, spacing, spacing)

        self.envs = []
        self.actors = []
        self.quads = []

        self._load_assets()

        for i in range(self.num_envs):
            env = self.gym.create_env(
                self.sim, self.env_lower_bound, self.env_upper_bound, envs_per_row
            )
            self.actors.append([])
            self.envs.append(env)
            self._create_env(i)

    def compute_observations(self):
        self.obs_buf[..., 0] = (TARGET_X - self.root_positions[..., 0, 0]) / 3
        self.obs_buf[..., 1] = (TARGET_Y - self.root_positions[..., 0, 1]) / 3
        self.obs_buf[..., 2] = (TARGET_Z - self.root_positions[..., 0, 2]) / 3
        self.obs_buf[..., 3:7] = self.root_rotations[..., 0, :]
        self.obs_buf[..., 7:10] = self.root_linear_velocities[..., 0, :] / 2
        self.obs_buf[..., 10:13] = self.root_angular_velocities[..., 0, :] / torch.pi
        return self.obs_buf

    def compute_reward(self):
        (
            reward,
            pos_reward,
            up_reward,
            spinnage_reward,
            vel_reward,
            target_dist,
            reset,
        ) = self.compute_quadcopter_reward_fn(
            self.target_tensor,
            self.root_positions[..., 0, :],
            self.root_rotations[..., 0, :],
            self.root_linear_velocities[..., 0, :],
            self.root_angular_velocities[..., 0, :],
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length,
        )

        self.rew_buf = reward
        self.reset_buf = reset

        if self.wandb:
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


@torch.jit.script
def compute_quadcopter_reward(
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
    spinnage = torch.norm(root_angvels)
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
        target_dist > 3.5,
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
