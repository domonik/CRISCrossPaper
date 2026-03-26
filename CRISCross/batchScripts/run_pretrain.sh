#!/bin/bash

#SBATCH --job-name=PaperPretrain
#SBATCH --output=logs/PretrainPaper2/%x_%A_%a.out
#SBATCH --error=logs/PretrainPaper2/%x_%A_%a.err
#SBATCH --partition=bidlc2_gpu-l40s
#SBATCH --gres=gpu:4,localtmp:1000
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=4
#SBATCH --time=15:00:00
#SBATCH --export=ALL
#SBATCH --exclude dlc2gpu07,dlc2gpu09,dlc2gpu11

# Array settings
#SBATCH --array=0-5%6    
srun python -m src.pretrain --config configs/pretrain_configs.json