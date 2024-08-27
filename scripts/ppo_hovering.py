import isaacgym  # noqa: F401
import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer

from cps_project.tasks.quadrotor import Quadrotor
import yaml
import os
import argparse
import wandb

# This script implements a PPO agent for training a quadrotor to reach
# a target inside an Isaac Gym environment. The environment is created
# using the Quadrotor class from the cps_project package.


class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
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
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )

        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(128, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            return (
                self.mean_layer(self.net(inputs["states"])),
                self.log_std_parameter,
                {},
            )
        elif role == "value":
            return self.value_layer(self.net(inputs["states"])), {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to the configuration file (relative to the script directory)",
        default="../cps_project/cfg/Quadrotor.yaml",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.abspath(os.path.join(script_dir, args.config))

    with open(cfg_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    task = Quadrotor(
        cfg=cfg,
        rl_device="cuda:0",
        sim_device="cuda:0",
        graphics_device_id=0,
        headless=False,
        virtual_screen_capture=False,
        force_render=True,
    )

    env = wrap_env(task)
    device = env.device
    memory = RandomMemory(memory_size=8, num_envs=env.num_envs, device=device)

    # instantiate the agent's models (function approximators)
    models = {}
    models["policy"] = Shared(env.observation_space, env.action_space, device)
    models["value"] = models["policy"]  # same instance: shared model

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
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    wandb.init(
        project="cps-project",
        config={
            "learning_rate": cfg["learning_rate"],
            "epochs": cfg["learning_epochs"],
        },
    )

    trainer.train()
    agent.save("ppo_hovering.pt")

    wandb.finish()

    agent.load("ppo_hovering.pt")
    trainer.eval()


if __name__ == "__main__":
    main()
