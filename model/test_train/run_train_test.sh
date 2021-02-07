#!/bin/bash
#SBATCH --account=mlclouds
#SBATCH --output=stdout_%A.txt
#SBATCH --error=stdout_%A.txt
#SBATCH --time=240
#SBATCH --qos=high

python train_n_test.py
