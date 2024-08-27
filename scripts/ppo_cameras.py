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
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from cps_project.tasks.quadrotor import Quadrotor
import yaml
import os
import argparse

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


class Critic(DeterministicMixin, Model):
    def __init__(
        self,
        drone_states_size,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        image_resolution=None,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.image_resolution = image_resolution

        self.net = nn.Sequential(
            nn.Linear(drone_states_size, 256),
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

        split_index = self.image_resolution["height"] * self.image_resolution["width"]
        critic_x = inputs["states"][:, split_index:]

        return self.net(critic_x), {}


class Actor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        image_resolution,
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

        self.actor_net = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
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
            # 67 is the height of the image, 120 is the width
            nn.Linear(
                32
                * (self.image_resolution["height"] // 4)
                * (self.image_resolution["width"] // 4),
                self.num_actions,
            ),
        )

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # view (samples, width * height * channels) -> (samples, width, height, channels)
        # permute (samples, width, height, channels) -> (samples, channels, width, height)
        # (samples x width * height + 13) -> [(samples, width, height, channels), (samples, 13)]

        split_index = self.image_resolution["height"] * self.image_resolution["width"]
        image_tensor = inputs["states"][:, 0:split_index]

        actor_x = image_tensor.view(
            -1, self.image_resolution["height"], self.image_resolution["width"]
        )
        actor_x = actor_x.unsqueeze(1)
        actor_x.permute(0, 1, 3, 2)

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

    drone_states_size = 13

    env = wrap_env(task)

    device = env.device

    # instantiate a memory as rollout buffer (any memory can be used for this)
    memory = RandomMemory(memory_size=8, num_envs=env.num_envs, device=device)

    # instantiate the agent's models (function approximators).
    # PPO requires 2 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
    models = {}
    # 13 is the size of the state of the drone because: positions x-y-z (3) + orientation quaternion (4) + velocity x-y-z (3) + angular velocity x-y-z (3)
    models["policy"] = Actor(
        env.observation_space,
        env.action_space,
        device,
        image_resolution=cfg["env"]["image"]["resolution"],
    )
    # models["value"] = models["policy"]  # same instance: shared model
    models["value"] = Critic(
        drone_states_size,
        env.observation_space,
        env.action_space,
        device,
        clip_actions=False,
        image_resolution=cfg["env"]["image"]["resolution"],
    )

    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 8  # memory_size
    cfg["learning_epochs"] = 8
    cfg["mini_batches"] = 4  # 8 * 8192 / 16384
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

    cfg["experiment"]["wandb"] = True  # enable wandb logging

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 8000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="cps-project",
    #     # track hyperparameters and run metadata
    #     config={
    #         "learning_rate": cfg["learning_rate"],
    #         "epochs": cfg["learning_epochs"],
    #     },
    # )

    # start training
    # trainer.train()

    # save the model
    agent.save("agent_pos_and_vel_reward.pt")

    # wandb.finish()

    # ---------------------------------------------------------
    # comment the code above: `trainer.train()`, and...
    # uncomment the following lines to evaluate a trained agent
    # ---------------------------------------------------------
    # from skrl.utils.huggingface import download_model_from_huggingface

    # # download the trained agent's checkpoint from Hugging Face Hub and load it
    # path = download_model_from_huggingface("skrl/IsaacGymEnvs-Quadcopter-PPO", filename="agent.pt")
    agent.load("agent_pos_and_vel_reward.pt")

    # start evaluation
    trainer.eval()


if __name__ == "__main__":
    main()
