# CPS Project

This repository contains source code, resources, documentation and final results of the project for the Cyber-Physical Systems course of Automation Engineering at the Univerity of Bologna for the academic year 2023/2024.

## Software dependecies & Setup

To run the code inside the repo you need to have [Docker](https://docs.docker.com/get-started/) installed on your machine. The code has been tested with the version 26.1.4 (build 5650f9b) of Docker on an Ubuntu 22.04.4 LTS machine with the following specifications:
```bash
PC Model: ASUS TUF Gaming fx 505 ge
CPU: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
GPU: NVIDIA GeForce GTX 1050 Ti - 4GB (Driver Version: 535.183.01, CUDA Version: 12.2)
RAM: 16GB
```

To run the code, the NVIDIA Container Toolkit must be installed following the instructions available [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Also, the steps in the [configuration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker) part of the documentation must be followed.

Check the configuration by running `cat /etc/docker/daemon.json`. The output should be similar to the one below:
```json
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```

To test that your setup is working correctly, try to run a sample workload following the official instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html#running-a-sample-workload-with-docker).

If everything is working correctly, you can proceed to install Isaac Gym by downloading the [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym/download) package.  Unzip the package in a folder of your choice and clone the [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) repo inside it. Then, edit the `docker/Dockerfile` file to replace the dependencies installation command with the following:
```Dockerfile
# dependencies for gym
#
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
 gcc-8 \
 g++-8 \
 vulkan-utils \
 mesa-vulkan-drivers \
 pigz \
 git \
 libegl1 \
 git-lfs \
 libsm6 \
 libxext6 \
 libxrender1 
```

Then, run the following command from the root of the folder to build the Isaac Gym Docker image:
```bash
bash docker/build.sh
```

After the image is built, you have to replace the content of the `docker/run.sh` file within the Isaac Gym folder with the following code (be careful to replace `/path/to/cps_project_folder` with the path to the folder containing the code of this repository)

```bash
#!/bin/bash
set -e
set -u

export DISPLAY=$DISPLAY
echo "setting display to $DISPLAY"

docker run -dit -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --gpus=all --name=isaacgym_container --volume=/path/to/cps_project_folder:/opt/isaacgym/cps_project isaacgym 
```

Then, you can run the Isaac Gym container by running the following command from the root of the Isaac Gym folder:
```bash
bash docker/run.sh
```

Now, you should have a container running with Isaac Gym installed and the code of this repository mounted inside it. If this is the case, you can access it by running the following command:
```bash
docker exec -it isaacgym_container bash
```

As a final step, you need to install the `cps_project` package and its Python dependencies by running the following command from the root of the mounted folder:
```bash
pip install -e .
```

Finally, you need to install the `IsaacGymEnvs` package by running the following command inside `/opt/isaacgym/IsaacGymEnvs`:
```bash
pip install -e .
``` 

## Usage

The code is organized in the following way:
- `cps_project` package: contains the source code of the project as long as the Isaac Gym tasks
- `resources` folder: contains the urdf models for the Isaac Gym environments
- `scripts` folder: contains the scripts to run the training and evaluation of the models
- `docs` folder: contains the Isaac Gym documentation, some useful resources and the final PowerPoint presentation of the project

To see an example of the environment in which the training is performed, run the following command from the root of the repository (inside the Isaac Gym container):
```bash
python scripts/basic_env.py
```

To run the training for the hovering task, without the obstacles avoidance part, run (always from the root of the repository, inside the Isaac Gym container):
```bash
python scripts/ppo_hovering.py
```

To run the training with cameras and obstacles, run (always from the root of the repository, inside the Isaac Gym container):
```bash
python scripts/ppo_cameras.py
```

By default, wandb is disabled. To enable it, add the flag ```--wandb``` and then log in with your account.

## Known Issues

- It is common that running the simulation for the first time, the following error is obtained:
    ```bash
    [Error] [carb.gym.plugin] Gym cuda error: an illegal memory access was encountered: ../../../source/plugins/carb/gym/impl/Gym/GymPhysX.cpp: 1373
    ```

    Simply re-running the script, the error will not be present anymore.

- A "Not Responding" window for "Isaac Gym" may occasionally pop up, particularly when using wandb. However, the simulation is still running, so there's no need to click "Force Quit." Instead, just select "Wait."

## Useful Resources & Links

Here we have collected and categorized various resources that have been useful during the project development

### Deep Learning, NN, CNN

Links:
- [CNN with MNIST 1](https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/)
- [CNN with MNIST 2](https://www.kaggle.com/code/sdelecourt/cnn-with-pytorch-for-mnist)
- [CNN with MNIST 3](https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118)
- [CNN theory](https://cs231n.github.io/convolutional-networks/)

Books:
- Deep Reinforcement Learning Hands-On: Apply Modern RL Methods, Maxim Lapan
- Foundations of Deep Reinforcement Learning: Theory and Practice in Python

### Useful repos

- [Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator) 





