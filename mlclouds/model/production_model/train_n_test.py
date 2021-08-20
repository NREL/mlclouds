"""
Randomly split full dataset using 80% for training and 20% for testing
"""
import os
import numpy as np
import json
from mlclouds.trainer import Trainer
from mlclouds.validator import Validator
from rex.utilities.loggers import init_logger

init_logger('mlclouds', log_level='DEBUG', log_file=None)
init_logger('phygnn', log_level='INFO', log_file=None)

fp_config = './config.json'
with open(fp_config, 'r') as f:
    config = json.load(f)

out_dir = './outputs/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

fp_base = ('/projects/mlclouds/data_surfrad_9/{y}_{ew}_adj/'
           'mlclouds_surfrad_{y}.h5')
files = [fp_base.format(y=y, ew=ew) for y in range(2016, 2020)
         for ew in ('east', 'west')]
files_e = [fp_base.format(y=y, ew=ew) for y in range(2016, 2020)
           for ew in ('east', )]
files_w = [fp_base.format(y=y, ew=ew) for y in range(2016, 2020)
           for ew in ('west', )]

print('Number of files:', len(files))
print('Number of east files:', len(files_e))
print('Number of west files:', len(files_w))
print('Source files:', files)
print('Full config:', config)


fp_history = os.path.join(out_dir, 'training_history.csv')
fp_model = os.path.join(out_dir, 'mlclouds_model.pkl')
fp_stats = os.path.join(out_dir, 'validation_stats.csv')
fp_stats_e = os.path.join(out_dir, 'validation_stats_east.csv')
fp_stats_w = os.path.join(out_dir, 'validation_stats_west.csv')
file_iter = (files, files_e, files_w)
fp_iter = (fp_stats, fp_stats_e, fp_stats_w)


t = Trainer(train_sites='all', train_files=files, config=config,
            test_fraction=0.2)

t.model.history.to_csv(fp_history)
t.model.save_model(fp_model)

for val_files, fp_stats_out in zip(file_iter, fp_iter):
    file_mask = np.isin(t.train_data.observation_sources, val_files)
    test_set_mask = t.test_set_mask.copy()[file_mask]
    print('Validating on {} out of {} observations'
          .format(test_set_mask.sum(), len(test_set_mask)))
    v = Validator(t.model, config=config, val_files=val_files,
                  save_timeseries=False,
                  test_set_mask=test_set_mask)
    v.stats.to_csv(fp_stats_out)
