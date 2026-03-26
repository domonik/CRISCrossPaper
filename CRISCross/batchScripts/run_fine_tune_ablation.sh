#!/bin/bash

#SBATCH --job-name=crispretrain
#SBATCH --output=logs/FineTunePaper/%x_%a.out
#SBATCH --error=logs/FineTunePaper/%x_%a.err
#SBATCH --partition=bidlc2_gpu-l40s
#SBATCH --gres=gpu:1,localtmp:100
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --export=ALL
#SBATCH --requeue
#SBATCH --exclude dlc2gpu07,dlc2gpu10,dlc2gpu09
# Array settings
#SBATCH --array=0-33%16    # adjust 11 to (num_configs - 1)
source /home/rabsch/.bashrc
miniforge activate CRISPRoff

python -m src.main --config configs/run_configs_ablation_l40s.json