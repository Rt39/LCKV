#!/bin/bash

#SBATCH --job-name=GPU
#SBATCH --partition=normal
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mail-type=end
#SBATCH --mail-user=hwwysong@gmail.com
#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err
#SBATCH --time 2-12:00:00

HF_ENDPOINT=https://hf-mirror.com

bash run_h2o.sh