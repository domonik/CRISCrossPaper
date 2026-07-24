#!/bin/bash

#SBATCH --job-name=JoinPretrain
#SBATCH --output=logs/CRISPRATAGJoin/%x_%A_%a.out
#SBATCH --error=logs/CRISPRATAGJoin/%x_%A_%a.err
#SBATCH --partition=bidlc2_gpu-l40s
#SBATCH --gres=gpu:8,localtmp:200
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8


# Array settings
#SBATCH --array=1-10%1   # adjust 11 to (num_configs - 1)


source /home/rabsch/.bashrc
miniforge activate CRISPRoff

srun python -m src.pretrainArtificial --config configs/artificial_AG_param_JoinMethods.json