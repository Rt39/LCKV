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
#SBATCH -c1
#SBATCH --mem 100M

# TRANSFORMERS_OFFLINE=1 LCKV_INFERENCE=1 HF_ENDPOINT=https://hf-mirror.com NUMEXPR_MAX_THREADS=64 accelerate launch test_harness.py
# TRANSFORMERS_OFFLINE=1 LCKV_INFERENCE=1 HF_ENDPOINT=https://hf-mirror.com NUMEXPR_MAX_THREADS=64 python test_harness.py

bash run_h2o_generation.sh