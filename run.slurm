#!/bin/bash

#SBATCH --job-name=GPU
#SBATCH --partition=normal
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=hwwysong@gmail.com
#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err
#SBATCH --time 2-12:00:00

HF_ENDPOINT=https://hf-mirror.com
TRANSFORMERS_OFFLINE=1

bash run_h2o_generation.sh