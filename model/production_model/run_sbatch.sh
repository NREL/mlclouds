#!/bin/bash
# Run cross validation against all sites on eagle

rm -r ./outputs
rm -r ./stdout
mkdir ./outputs
mkdir ./stdout

echo Running production model train and test
sbatch run_train_test.sh
