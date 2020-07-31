"""
Run cross validation against a single site passed from the command line. This
is intended to be run from:

    sbatch single_axv.sh X

Where X is the site to validate against. Statistics are written to disk.
"""
from mlclouds.autoxval import AutoXVal, CONFIG
from rex.utilities.loggers import init_logger
import sys

init_logger('mlclouds', log_level='INFO', log_file=None)
init_logger('phygnn', log_level='INFO', log_file=None)

# Model configuration and hyperparameters can be changed by editing the CONFIG
# dict
config = CONFIG
config['n_epoch'] = 100
config['learning_rate'] = 0.011

print('Cross validating against ', sys.argv[1])

axv = AutoXVal(sites=[0, 1, 2, 3, 4, 5, 6], years=[2018, 2019],
               val_sites=str(sys.argv[1]),
               config=config, catch_nan=True)
