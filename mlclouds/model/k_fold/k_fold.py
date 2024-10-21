"""
K-fold train and validate with a single site left out of the training pool
"""

import json
import os
import sys

from rex.utilities.loggers import init_logger

from mlclouds.trainer import Trainer
from mlclouds.validator import Validator

init_logger('mlclouds', log_level='DEBUG', log_file=None)
init_logger('phygnn', log_level='INFO', log_file=None)

val_site = int(sys.argv[1])
print('Cross validation site: {}'.format(val_site))

fp_config = './config.json'
out_dir = './outputs'

with open(fp_config, 'r') as f:
    config = json.load(f)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

years = range(2016, 2020)
fp_base = (
    '/projects/pxs/mlclouds/training_data/{y}_{ew}_v322/'
    'mlclouds_surfrad_{ew}_{y}.h5'
)
files = [fp_base.format(y=y, ew=ew) for y in years for ew in ('east', 'west')]
files_e = [fp_base.format(y=y, ew=ew) for y in years for ew in ('east',)]
files_w = [fp_base.format(y=y, ew=ew) for y in years for ew in ('west',)]

train_sites = [i for i in range(9) if i != val_site]

print('Training sites: {}'.format(train_sites))
print('Number of files:', len(files))
print('Number of east files:', len(files_e))
print('Number of west files:', len(files_w))
print('Source files:', files)
print('Full config:', config)

fp_history = os.path.join(out_dir, 'training_history_{}.csv'.format(val_site))
fp_model = os.path.join(out_dir, 'model_{}.pkl'.format(val_site))
fp_stats = os.path.join(out_dir, 'validation_stats_{}.csv'.format(val_site))
fp_stats_e = os.path.join(
    out_dir, 'validation_stats_east_{}.csv'.format(val_site)
)
fp_stats_w = os.path.join(
    out_dir, 'validation_stats_west_{}.csv'.format(val_site)
)
file_iter = (files, files_e, files_w)
fp_iter = (fp_stats, fp_stats_e, fp_stats_w)

if __name__ == '__main__':
    t = Trainer(train_sites=train_sites, train_files=files, config=config)

    t.model.history.to_csv(fp_history)
    t.model.save_model(fp_model)

    for val_files, fp_stats_out in zip(file_iter, fp_iter):
        v = Validator(
            t.model, config=config, val_files=val_files, save_timeseries=False
        )
        v.stats.to_csv(fp_stats_out)
