#!/bin/bash

#SBATCH --job-name=ArtiFineTune
#SBATCH --output=logs/FineTunePaperArti/%x_%a.out
#SBATCH --error=logs/FineTunePaperArti/%x_%a.err
#SBATCH --partition=bidlc2_gpu-l40s
#SBATCH --gres=gpu:1,localtmp:100
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --export=ALL
#SBATCH --requeue
#SBATCH --exclude dlc2gpu07,dlc2gpu10,dlc2gpu09
# Array settings
#SBATCH --array=45-57%18    # adjust 11 to (num_configs - 1)

source /home/rabsch/.bashrc
miniforge activate CRISPRoff

python -m src.fineTuning --config configs/run_artifical_configs_l40s.json