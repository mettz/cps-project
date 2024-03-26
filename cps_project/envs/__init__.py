# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from cps_project.envs.base.aerial_robot_config import AerialRobotCfg
from cps_project.envs.base.aerial_robot_with_obstacles_config import (
    AerialRobotWithObstaclesCfg,
)
from cps_project.utils.task_registry import task_registry

from .base.aerial_robot import AerialRobot
from .base.aerial_robot_with_obstacles import AerialRobotWithObstacles

task_registry.register("quad", AerialRobot, AerialRobotCfg())
task_registry.register(
    "quad_with_obstacles", AerialRobotWithObstacles, AerialRobotWithObstaclesCfg()
)
