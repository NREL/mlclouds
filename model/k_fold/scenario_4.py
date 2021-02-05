# from mlclouds.autoxval import AutoXVal, ValidationData, ALL_SKY_VARS
import json
import sys
from mlclouds.model_agents import Trainer, Validator
from rex.utilities.loggers import init_logger

init_logger('mlclouds', log_level='DEBUG', log_file=None)
init_logger('phygnn', log_level='INFO', log_file=None)

with open('./xval_config.json', 'r') as f:
    config = json.load(f)

east_2016 = '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2016' +\
                '_east_adj/mlclouds_surfrad_2016.h5'
east_2017 = '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2017' +\
                '_east_adj/mlclouds_surfrad_2017.h5'
east_2018 = '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2018' +\
                '_east_adj/mlclouds_surfrad_2018_adj.h5'
east_2019 = '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2019' +\
                '_east_adj/mlclouds_surfrad_2019_adj.h5'

west_2016 = '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2016' +\
                '_west_adj/mlclouds_surfrad_2016.h5'
west_2017 = '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2017' +\
                '_west_adj/mlclouds_surfrad_2017.h5'
west_2018 = '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2018' +\
                '_west_adj/mlclouds_surfrad_2018.h5'
west_2019 = '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2019' +\
                '_west_adj/mlclouds_surfrad_2019.h5'

# 4 Satellite + Temporal + Spatial Xval
val_site = int(sys.argv[1])
files = [east_2016, east_2017, east_2018, east_2019, west_2016, west_2017,
         west_2018, west_2019]
sites = [0, 1, 2, 3, 4, 5, 6, 7, 8]
train_sites = [x for x in sites if x != val_site]
config['timeseries_dir'] = 'ts_val_site_{}'.format(val_site)

if val_site in [6, 3, 4, 2]:
    config['learning_rate'] = 0.0005
else:
    config['learning_rate'] = 0.0010

print('config:', config)
print ('files:', files)

print ('val_site:', val_site)
print ('train_sites:', train_sites)

t = Trainer(train_sites=train_sites, train_files=files, config=config)
v = Validator(t.model, config=config, val_files=files, save_timeseries=True)
v.stats.to_csv('stats/stats_scenario_4_val_site_{}.csv'.format(val_site))
