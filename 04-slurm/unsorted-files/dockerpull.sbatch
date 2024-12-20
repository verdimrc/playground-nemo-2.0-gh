#!/bin/bash

# Parameters
#SBATCH --exclusive
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=debug
#SBATCH --time=01:00:00

srun -l docker pull nvcr.io/nvidia/nemo:24.09
