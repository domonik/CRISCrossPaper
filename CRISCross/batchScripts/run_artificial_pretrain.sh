#!/bin/bash

#SBATCH --job-name=Artcrispretrain
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=bidlc2_gpu-h200
#SBATCH --gres=gpu:2,localtmp:200
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --export=ALL


# Array settings
#SBATCH --array=0-4%1   # adjust 11 to (num_configs - 1)


source /home/rabsch/.bashrc
miniforge activate CRISPRoff

python -m src.pretrainArtificial --config configs/artificial_param_combinations.json