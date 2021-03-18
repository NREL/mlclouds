#!/bin/bash
#SBATCH --account=pxs
#SBATCH --output=./stdout/stdout_%A.txt
#SBATCH --error=./stdout/stdout_%A.txt
#SBATCH --time=240
#SBATCH --qos=high

echo Starting scenario 4, validation against site $1
python k_fold.py $1
