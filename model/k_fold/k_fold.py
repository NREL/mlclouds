"""
K-fold train and validate with a single site left out of the training pool
"""
import os
import json
import sys
from mlclouds.trainer import Trainer
from mlclouds.validator import Validator
from rex.utilities.loggers import init_logger

init_logger('mlclouds', log_level='DEBUG', log_file=None)
init_logger('phygnn', log_level='INFO', log_file=None)

val_site = int(sys.argv[1])
print('Cross validation site: {}'.format(val_site))

fp_config = './config_optm.json'
out_dir = './outputs'

with open(fp_config, 'r') as f:
    config = json.load(f)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

fp_base = ('/projects/mlclouds/data_surfrad_9/{y}_{ew}_adj/'
           'mlclouds_surfrad_{y}.h5')
files = [fp_base.format(y=y, ew=ew)
         for y in range(2016, 2020)
         for ew in ('east', 'west')]

train_sites = [i for i in range(9) if i != val_site]

print('Training sites: {}'.format(train_sites))
print('Number of files:', len(files))
print('Source files:', files)
print('Full config:', config)

fp_history = os.path.join(out_dir, 'training_history_{}.csv'.format(val_site))
fp_model = os.path.join(out_dir, 'model_{}.pkl'.format(val_site))
fp_stats_out = os.path.join(out_dir, 'stats_k_fold_{}.csv'.format(val_site))

t = Trainer(train_sites=train_sites, train_files=files, config=config)

t.model.history.to_csv(fp_history)
t.model.save_model(fp_model)

v = Validator(t.model, config=config, val_files=files, save_timeseries=False)
v.stats.to_csv(fp_stats_out)
