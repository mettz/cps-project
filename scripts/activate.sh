#!/bin/bash

export LD_LIBRARY_PATH="/home/danielepc/miniconda3/envs/rlgpu/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/danielepc/miniconda3/envs/rlgpu/lib/python3.7/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH"

conda activate rlgpu