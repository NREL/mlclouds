"""
Run cross validation against a single site passed from the command line. This
is intended to be run from:

    sbatch single_axv.sh X

Where X is the site to validate against. Statistics are written to disk.
"""
from mlclouds.autoxval import AutoXVal, CONFIG
from rex.utilities.loggers import init_logger
import sys
from datetime import datetime as dt

init_logger('mlclouds', log_level='INFO', log_file=None)
init_logger('phygnn', log_level='INFO', log_file=None)

# Model configuration and hyperparameters can be changed by editing the CONFIG
# dict
config = CONFIG
config['epochs_a'] = 100
config['epochs_b'] = 100
config['learning_rate'] = 0.011

print(dt.now(), 'Cross validating against', sys.argv[1])

# K-fold validation - only train on n-1 sites, plots in jupyter notebook
# will not work correctly
if False:
    axv = AutoXVal.k_fold(sites=[0, 1, 2, 3, 4, 5, 6], years=[2018, 2019],
                          val_sites=int(sys.argv[1]), config=config,
                          catch_nan=True)

# K by N validation - train on 1, 2, ..., n - 1 sites, saving all results
# The jupyter notebook can be used to plot MAE % versus number of training
# sites
if True:
    axv = AutoXVal.kxn_fold(sites=[0, 1, 2, 3, 4, 5, 6], years=[2018, 2019],
                            val_sites=int(sys.argv[1]), config=config,
                            catch_nan=True)
