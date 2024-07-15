#!/bin/bash

#SBATCH --job-name=GPU
#SBATCH --partition=normal
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mail-type=end
#SBATCH --mail-user=hwwysong@gmail.com
#SBATCH --output=%j.out
#SBATCH --error=%j.err

HF_ENDPOINT=https://hf-mirror.com

bash run_h2o.sh