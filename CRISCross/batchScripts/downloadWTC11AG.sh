#!/bin/bash

#SBATCH --job-name=downloadAG
#SBATCH --output=logs/downloadAGWTC/%x.out
#SBATCH --error=logs/downloadAGWTC/%x.err
#SBATCH --partition=bidlc2_gpu-l40s
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --export=ALL
#SBATCH --requeue



# Run your Python script
PYTHONPATH=./ python util/downloadWholeAGTrack.py --ontology "EFO:0009747"
