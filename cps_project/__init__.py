# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

cps_project_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
cps_project_ENVS_DIR = os.path.join(cps_project_ROOT_DIR, "cps_project", "envs")

print("cps_project_ROOT_DIR", cps_project_ROOT_DIR)
