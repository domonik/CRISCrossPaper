#!/bin/bash

#SBATCH --job-name=TestCRISPRAT
#SBATCH --output=logs/FineTuneAndTestCRISPRATAblation/%x_%a.out
#SBATCH --error=logs/FineTuneAndTestCRISPRATAblation/%x_%a.err
#SBATCH --partition=bidlc2_gpu-l40s
#SBATCH --gres=gpu:1,localtmp:100
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --export=ALL
#SBATCH --requeue
#SBATCH --exclude dlc2gpu07,dlc2gpu10,dlc2gpu09
# Array settings
#SBATCH --array=0-99%18

source /home/rabsch/.bashrc
miniforge activate CRISPRoff

python -m src.fineTuning --config configs/fineTuningCRISPRATAblation.json
python -m src.fineTuning --config configs/TestCRISPRATAblation.json