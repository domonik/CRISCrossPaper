#!/bin/bash

#SBATCH --job-name=downloadAG
#SBATCH --output=logs/downloadAGK562/%x.out
#SBATCH --error=logs/downloadAGHK562/%x.err
#SBATCH --partition=bidlc2_gpu-l40s
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --export=ALL
#SBATCH --requeue



# Run your Python script
python -m util/downloadWholeAGTrack.py --ontology "EFO:0002067"
