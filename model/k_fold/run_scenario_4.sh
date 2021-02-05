#!/bin/bash
#SBATCH --account=mlclouds
#SBATCH --output=output/output_%A.txt
#SBATCH --error=output/output_%A.txt
#SBATCH --time=240
#SBATCH --qos=high

echo Starting scenario 4, validation against site $1
python scenario_4.py $1 
