#!/bin/bash
#SBATCH --job-name=agdownload
#SBATCH --output=agdlogs/%x_%A_%a.out
#SBATCH --error=agdlogs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH -p bidlc2_gpu-l40s        # Partition

#SBATCH --time=24:00:00           # Max runtime (adjust as needed)
#SBATCH --export=ALL


python util/get_AGfeatures.py