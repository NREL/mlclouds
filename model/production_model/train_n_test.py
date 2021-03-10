"""
Randomly split full dataset using 80% for training and 20% for testing
"""
import os
import json
from mlclouds.autoxval import TrainTest
from rex.utilities.loggers import init_logger

init_logger('mlclouds', log_level='DEBUG', log_file=None)
init_logger('phygnn', log_level='INFO', log_file=None)

fp_config = './config_optm.json'
with open(fp_config, 'r') as f:
    config = json.load(f)

out_dir = './outputs/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

fp_base = ('/projects/mlclouds/data_surfrad_9/{y}_{ew}_adj/'
           'mlclouds_surfrad_{y}.h5')
files = [fp_base.format(y=y, ew=ew)
         for y in range(2016, 2020)
         for ew in ('east', 'west')]

print('Number of files:', len(files))
print('Source files:', files)
print('Full config:', config)

tt = TrainTest(files, config=config,
               stats_file=os.path.join(out_dir, 'validation_stats.csv'),
               history_file=os.path.join(out_dir, 'training_history.csv'),
               model_file=os.path.join(out_dir, 'mlclouds_model.pkl'))
