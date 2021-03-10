#!/bin/bash
# Run cross validation against all sites on eagle

rm -r ./outputs
rm -r ./stdout
mkdir ./outputs
mkdir ./stdout

for SITE in 0 1 2 3 4 5 6 7 8
do
    echo Kicking off kfold for validation against site number $SITE
    sbatch run_k_fold_single.sh $SITE
    sleep 1
done
