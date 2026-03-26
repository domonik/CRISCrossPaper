#!/bin/bash

#SBATCH --job-name=downloadAG
#SBATCH --output=logs/downloadAGTCELL/%x.out
#SBATCH --error=logs/downloadAGTCELL/%x.err
#SBATCH --partition=bidlc2_gpu-l40s
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --requeue



# Run your Python script
python -m util/downloadWholeAGTrack.py --ontology "CL:0000624"
