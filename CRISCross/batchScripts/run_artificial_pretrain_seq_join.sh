#!/bin/bash

#SBATCH --job-name=JoinSeqPretrain
#SBATCH --output=logs/CRISPRATSeqJoin/%x_%A_%a.out
#SBATCH --error=logs/CRISPRATSeqJoin/%x_%A_%a.err
#SBATCH --partition=bidlc2_gpu-l40s
#SBATCH --gres=gpu:8,localtmp:200
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --exclude=dlc2gpu18


# Array settings
#SBATCH --array=1-10%1   # adjust 11 to (num_configs - 1)

nvidia-smi
python - <<EOF
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
EOF

srun python -m src.pretrainArtificial --config configs/artificial_SEQ_param_JoinMethods.json