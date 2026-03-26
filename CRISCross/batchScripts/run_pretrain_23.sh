#!/bin/bash

#SBATCH --job-name=PaperPretrain
#SBATCH --output=logs/PretrainPaper2/%x_%A_%a.out
#SBATCH --error=logs/PretrainPaper2/%x_%A_%a.err
#SBATCH --partition=bidlc2_gpu-l40s
#SBATCH --gres=gpu:1,localtmp:300
#SBATCH --cpus-per-task=15
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --exclude dlc2gpu07,dlc2gpu09




# Array settings
#SBATCH --array=6-8%3    # adjust 11 to (num_configs - 1)

source /home/rabsch/.bashrc
miniforge activate CRISPRoff

srun python -m src.pretrain --config configs/pretrain_configs.json