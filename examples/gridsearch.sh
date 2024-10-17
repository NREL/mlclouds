#/bin/bash

python ../mlclouds/grid_searcher.py ../hyperparameter_tuning/gridsearch_default.json --conda_env=mlclouds_gpu --output_ws=/tmp --walltime=4
