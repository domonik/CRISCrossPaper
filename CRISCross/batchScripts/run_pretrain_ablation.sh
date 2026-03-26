#!/bin/bash

#SBATCH --job-name=AblationPretrain
#SBATCH --output=logs/AblationPretrain/%x_%A_%a.out
#SBATCH --error=logs/AblationPretrain/%x_%A_%a.err
#SBATCH --partition=bidlc2_gpu-l40s
#SBATCH --gres=gpu:4,localtmp:100
#SBATCH --cpus-per-task=15
#SBATCH --ntasks-per-node=4
#SBATCH --time=20:00:00
#SBATCH --export=ALL
#SBATCH --exclude dlc2gpu07,dlc2gpu09
#SBATCH --mem=256GB




# Array settings
#SBATCH --array=0-1%10    # adjust 11 to (num_configs - 1)

source /home/rabsch/.bashrc
miniforge activate CRISPRoff

srun python -m src.pretrain --config configs/ablation_pretrain_configs.json