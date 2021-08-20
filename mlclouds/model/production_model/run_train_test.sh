#!/bin/bash
#SBATCH --account=pxs
#SBATCH --output=./stdout/stdout_%A.txt
#SBATCH --error=./stdout/stdout_%A.txt
#SBATCH --time=60
#SBATCH -p debug
#SBATCH --qos=high
python train_n_test.py