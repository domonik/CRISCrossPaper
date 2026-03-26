#!/bin/bash

#SBATCH --job-name=CRISPRATAG
#SBATCH --output=logs/CRISPRATAG/%x_%A_%a.out
#SBATCH --error=logs/CRISPRATAG/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=bidlc2_gpu-h200
#SBATCH --gres=gpu:2,localtmp:400
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --export=ALL


# Array settings
#SBATCH --array=0-4%1   # adjust 11 to (num_configs - 1)


source /home/rabsch/.bashrc
miniforge activate CRISPRoff

srun python -m src.pretrainArtificial --config configs/artificial_AG_param_combinations.json