"""
Randomly split full dataset using 80% for training and 20% for testing
"""
import json
from mlclouds.autoxval import TrainTest
from rex.utilities.loggers import init_logger

init_logger('mlclouds', log_level='DEBUG', log_file=None)
init_logger('phygnn', log_level='INFO', log_file=None)

with open('./xval_config.json', 'r') as f:
    config = json.load(f)
config["learning_rate"] = 0.0015
config["learning_rate"] = 0.001
config["learning_rate"] = 0.0001
config["learning_rate"] = 0.0021
config["learning_rate"] = 0.003

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

fs = {
      # 'east_files':  [east_2016, east_2017, east_2018, east_2019],
      'all_files': [east_2016, east_2017, east_2018, east_2019,
                    west_2016, west_2017, west_2018, west_2019],
      # 'west_files':  [west_2016, west_2017, west_2018, west_2019]
}

scenario = 'all_files'
files = fs[scenario]

test_fraction = 0.2

print('scenario:', scenario)
print('files:', files)
print('config:', config)
print('test_fraction:', test_fraction)

tt = TrainTest(files, config=config, test_fraction=test_fraction,
               stats_file='stats/stats_{}.csv'.format(scenario))
