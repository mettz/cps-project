import os

import numpy as np
import yaml
from isaacgym import gymutil
import torch
from skrl.envs.wrappers.torch import wrap_env

from cps_project.tasks.quadrotor import Quadrotor


def main():
    # TODO: check why this is needed
    """Setting seeds for random number generators is essential for reproducibility in computational experiments.
    When random number generators are initialized with the same seed, they produce the same sequence of numbers
    each time the code is run. This allows for consistent results, which is particularly important in machine learning and
    reinforcement learning tasks where randomness plays a significant role."""
    # direi che quindi in basic_env non serve

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args = gymutil.parse_arguments(
        description="Simple simulation to show basic drone environment with obstacles",
        custom_parameters=[
            {
                "name": "--envs",
                "type": int,
                "default": "256",
                "help": "Number of environments to create. Overrides config file if provided.",
            },
            {
                "name": "--config",
                "type": str,
                "default": "../cps_project/cfg/Quadrotor.yaml",
                "help": "Path to the configuration file. Overrides default configuration.",
            },
        ],
    )
    # Open yaml file and read the configuration

    # Combine the base directory with the relative path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.abspath(os.path.join(script_dir, args.config))
    print(f"Using configuration file: {cfg_path}")

    # Get the absolute path by combining it with the current working directory

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

    # env = wrap_env(task)
    # print("Environment created with device", env.device)

    # print("Number of environments", task.num_envs)

    # these are the actions commanded to the drone
    command_actions = torch.zeros((task.num_envs, task.num_actions))
    command_actions[:, 0] = 0.0  # velocity along x
    command_actions[:, 1] = 0.0  # velocity along y
    command_actions[:, 2] = 0.0  # velocity along z
    command_actions[:, 3] = -1.0  # yaw rate

    task.reset()
    i = 1
    print("Starting simulation. Press Ctrl+C to stop it.")
    for i in range(3000):
        print(f"Simulation step {i}", end="\r")
        task.step(command_actions)
        i += 1


if __name__ == "__main__":
    main()
