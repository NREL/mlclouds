"""
Randomly split full dataset using 80% for training and 20% for testing
"""
import json
from mlclouds.autoxval import TrainTest
from rex.utilities.loggers import init_logger

init_logger('mlclouds.data_cleaners', log_level='DEBUG', log_file=None)
init_logger('mlclouds.data_handlers', log_level='DEBUG', log_file=None)
init_logger('mlclouds.model_agents', log_level='DEBUG', log_file=None)
init_logger('phygnn', log_level='INFO', log_file=None)

fp_config = './config_optm.json'

with open(fp_config, 'r') as f:
    config = json.load(f)

fp_base = ('/projects/mlclouds/data_surfrad_9/{y}_{ew}_adj/'
           'mlclouds_surfrad_{y}.h5')

files = [fp_base.format(y=y, ew=ew)
         for y in range(2016, 2020)
         for ew in ('east', 'west')]

print('files:', files)
print('n_files:', len(files))
print('config:', config)

tt = TrainTest(files, config=config,
               stats_file='validation_stats.csv',
               history_file='training_history.csv')
