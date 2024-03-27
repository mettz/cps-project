from isaacgym import gymutil
import torch
import numpy as np
import os

from cps_project.envs import AerialRobotWithObstacles, AerialRobotWithObstaclesCfg
from cps_project.utils.helpers import parse_sim_params, class_to_dict


def main():
    # TODO: check why this is needed
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
            }
        ],
    )

    cfg = AerialRobotWithObstaclesCfg()
    sim_params = {"sim": class_to_dict(cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)

    task = AerialRobotWithObstacles(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
    )

    print("Number of environments", cfg.env.num_envs)

    # these are the actions commanded to the drone
    command_actions = torch.zeros((cfg.env.num_envs, cfg.env.num_actions))
    command_actions[:, 0] = 0.0  # velocity along x
    command_actions[:, 1] = 0.0  # velocity along y
    command_actions[:, 2] = 0.0  # velocity along z
    command_actions[:, 3] = 0.0  # yaw rate

    task.reset()
    i = 1
    print("Starting simulation. Press Ctrl+C to stop it.")
    while True:
        print(f"Simulation step {i}", end="\r")
        task.step(command_actions)
        i += 1


if __name__ == "__main__":
    main()
