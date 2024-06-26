"""
Train and test the MLClouds production model
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

years = range(2016, 2020)
fp_base = ('/projects/pxs/mlclouds/training_data/{y}_{ew}_v322/'
           'mlclouds_surfrad_{ew}_{y}.h5')
files = [fp_base.format(y=y, ew=ew) for y in years for ew in ('east', 'west')]
files_e = [fp_base.format(y=y, ew=ew) for y in years for ew in ('east', )]
files_w = [fp_base.format(y=y, ew=ew) for y in years for ew in ('west', )]


print('Number of files:', len(files))
print('Number of east files:', len(files_e))
print('Number of west files:', len(files_w))
print('Source files:', files)
print('Full config:', config)


fp_history = os.path.join(out_dir, 'training_history.csv')
fp_model = os.path.join(out_dir, 'mlclouds_model.pkl')
fp_env = os.path.join(out_dir, 'mlclouds_model_env.json')
fp_stats = os.path.join(out_dir, 'validation_stats.csv')
fp_stats_e = os.path.join(out_dir, 'validation_stats_east.csv')
fp_stats_w = os.path.join(out_dir, 'validation_stats_west.csv')
file_iter = (files, files_e, files_w)
fp_iter = (fp_stats, fp_stats_e, fp_stats_w)


if __name__ == '__main__':
    t = Trainer(train_sites='all', train_files=files, config=config,
                test_fraction=0.2, fp_save_data='./mlclouds_training_data.csv')

    t.model.history.to_csv(fp_history)
    t.model.save_model(fp_model)
    with open(fp_env, 'w') as f:
        json.dump(t.model.version_record, f)

    for val_files, fp_stats_out in zip(file_iter, fp_iter):
        file_mask = np.isin(t.train_data.observation_sources, val_files)
        test_set_mask = t.test_set_mask.copy()[file_mask]
        print('Validating on {} out of {} observations'
              .format(test_set_mask.sum(), len(test_set_mask)))
        v = Validator(t.model, config=config, val_files=val_files,
                      save_timeseries=False,
                      test_set_mask=test_set_mask)
        v.stats.to_csv(fp_stats_out)
