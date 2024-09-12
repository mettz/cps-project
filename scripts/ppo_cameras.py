import isaacgym
import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import ParallelTrainer

import wandb

from cps_project.tasks.quadrotor_cameras import QuadrotorCameras
import yaml
import os
import argparse

# This script implements a PPO agent for training a quadrotor to reach
# a target inside an Isaac Gym environment. The environment is created
# using the Quadrotor class from the cps_project package.


class Critic(DeterministicMixin, Model):
    def __init__(
        self,
        drone_states_size,
        observation_space,
        action_space,
        device,
        clip_actions=False,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.drone_states_size = drone_states_size

        self.net = nn.Sequential(
            nn.Linear(self.drone_states_size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, _):
        # view (samples, width * height * channels) -> (samples, width, height, channels)
        # permute (samples, width, height, channels) -> (samples, channels, width, height)
        # (samples x width * height + 13) -> [(samples, 13), (samples, width, height, channels)]

        critic_x = inputs["states"][:, -self.drone_states_size :]

        return self.net(critic_x), {}


class Actor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        image_resolution,
        camera_type="depth",
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction
        )

        self.image_resolution = image_resolution
        self.camera_type = camera_type

        self.actor_net = nn.Sequential(
            nn.Conv2d(
                in_channels=1 if camera_type == "depth" else 4,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(
                32
                * (self.image_resolution["height"] // 4)
                * (self.image_resolution["width"] // 4),
                self.num_actions,
            ),
        )

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, _):
        # view (samples, width * height * channels) -> (samples, width, height, channels)
        # permute (samples, width, height, channels) -> (samples, channels, width, height)
        # (samples x width * height + 13) -> [(samples, width, height, channels), (samples, 13)]

        split_index = self.image_resolution["height"] * self.image_resolution["width"]
        if self.camera_type != "depth":
            split_index *= 4

        image_tensor = inputs["states"][:, 0:split_index]  # 4

        if self.camera_type == "depth":
            actor_x = image_tensor.view(
                -1, self.image_resolution["height"], self.image_resolution["width"]
            )
            actor_x = actor_x.unsqueeze(1)
            actor_x.permute(0, 1, 3, 2)
        else:
            actor_x = image_tensor.reshape(
                -1,
                self.image_resolution["height"],
                self.image_resolution["width"],
                4,
            )
            actor_x = actor_x.permute(0, 3, 1, 2)

        return (self.actor_net(actor_x), self.log_std_parameter, {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to the configuration file",
        default="../cps_project/cfg/Quadrotor.yaml",
    )
    parser.add_argument(
        "--wandb",
        help="Use Weights & Biases for logging",
        action="store_true",
    )
    parser.add_argument(
        "--eval",
        help="Evaluate the trained agent",
        action="store_true",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.abspath(os.path.join(script_dir, args.config))

    with open(cfg_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    task = QuadrotorCameras(
        cfg=cfg,
        rl_device="cuda:0",
        sim_device="cuda:0",
        graphics_device_id=0,
        headless=False,
        virtual_screen_capture=False,
        force_render=True,
        wandb=args.wandb,
    )

    # 13 is the size of the state of the drone because: positions x-y-z (3) + orientation quaternion (4) + velocity x-y-z (3) + angular velocity x-y-z (3)
    drone_states_size = 13

    env = wrap_env(task)
    device = env.device
    memory = RandomMemory(memory_size=8, num_envs=env.num_envs, device=device)

    # instantiate the agent's models (function approximators).
    models = {}
    models["policy"] = Actor(
        env.observation_space,
        env.action_space,
        device,
        image_resolution=cfg["env"]["image"]["resolution"],
        camera_type=cfg["sim"]["camera"],
    )
    models["value"] = Critic(
        drone_states_size,
        env.observation_space,
        env.action_space,
        device,
        clip_actions=False,
    )

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 8  # memory_size
    cfg["learning_epochs"] = 8
    cfg["mini_batches"] = 4
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 1e-3
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016}
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.0
    cfg["value_loss_scale"] = 1.0
    cfg["kl_threshold"] = 0
    cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.1
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 20
    cfg["experiment"]["checkpoint_interval"] = 200
    cfg["experiment"]["directory"] = "runs/torch/Quadcopter"

    if args.wandb:
        cfg["experiment"]["wandb"] = True

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    cfg_trainer = {"timesteps": 8000, "headless": True}
    trainer = ParallelTrainer(cfg=cfg_trainer, env=env, agents=agent)

    if args.wandb:
        wandb.init(
            project="cps-project",
            config={
                "learning_rate": cfg["learning_rate"],
                "epochs": cfg["learning_epochs"],
            },
        )

    if not args.eval:
        trainer.train()
        agent.save("ppo_cameras.pt")

    if args.wandb:
        wandb.finish()

    if args.eval:
        agent.load("ppo_cameras.pt")
        trainer.eval()


if __name__ == "__main__":
    main()
