#!/bin/bash

#SBATCH --job-name=AblationPretrain
#SBATCH --output=logs/CrossCellMLM/%x_%A_%a.out
#SBATCH --error=logs/CrossCellMLM/%x_%A_%a.err
#SBATCH --partition=bidlc2_gpu-l40s
#SBATCH --gres=gpu:4,localtmp:100
#SBATCH --cpus-per-task=15
#SBATCH --ntasks-per-node=4
#SBATCH --time=20:00:00
#SBATCH --export=ALL
#SBATCH --exclude dlc2gpu07,dlc2gpu09
#SBATCH --mem=256GB





# Array settings
#SBATCH --array=0-3%4    # adjust 11 to (num_configs - 1)

source /home/rabsch/.bashrc
miniforge activate CRISPRoff

srun python -m src.pretrainMLM --config configs/crosscell_mlm_pretrain_configs.json