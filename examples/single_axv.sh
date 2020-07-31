#!/bin/bash
#SBATCH --account=mlclouds
#SBATCH --output=output/output_%A.txt
#SBATCH --error=output/errors_%A.txt
#SBATCH --time=240
#SBATCH --qos=high

python single $1
