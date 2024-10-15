"""Train and test the MLClouds production model"""

import argparse
import json
import logging
import os
from glob import glob

import numpy as np
from rex.utilities.loggers import init_logger

from mlclouds.trainer import Trainer
from mlclouds.validator import Validator

init_logger('mlclouds', log_level='DEBUG', log_file=None)
init_logger('phygnn', log_level='INFO', log_file=None)
init_logger('nsrdb', log_level='INFO', log_file=None)
init_logger(__name__, log_level='DEBUG', log_file=None)

logger = logging.getLogger(__name__)

surfrad_fps_default = (
    '/projects/pxs/mlclouds/training_data/{year}_{area}_v322/'
    'mlclouds_surfrad_{area}_{year}.h5'
)
nsrdb_fps_default = '/projects/pxs/mlclouds/training_data/*_v322/final/*.h5'

parser = argparse.ArgumentParser(description='Train and test MLClouds model.')
parser.add_argument(
    'config_dir',
    type=str,
    help="""Directory with config.json file specifying model configuration.
         Outputs will be written to an "outputs" subdirectory within the
         config_dir if out_dir is not specified.""",
)
parser.add_argument(
    '-surfrad_fps',
    type=str,
    default=surfrad_fps_default,
    help="""File pattern name for training data at surfrad measurement
         locations. Must have {year} and {area} format keys.""",
)
parser.add_argument(
    '-nsrdb_fps',
    type=str,
    default=nsrdb_fps_default,
    help="""File pattern name for nsrdb training data. This is only used to
         filter training data such that the sky_class matches the cloud type
         label in the surfrad training data. e.g. Training data is filtered
         when the sky_class determined from the NSRDB data is clear and
         training data cloud type is cloudy, or vice versa.""",
)
parser.add_argument(
    '-cache_fps',
    type=str,
    default=None,
    help="""File pattern for cached training data. Must have an empty format
         key '{}' which will be replaced with either "all_sky" or "raw" upon
         loading.""",
)
parser.add_argument(
    '-years',
    nargs='+',
    default=range(2016, 2020),
    help='Years to use for training data.',
)
parser.add_argument(
    '-sites',
    nargs='+',
    type=int,
    default='all',
    help='Site gids to use for training and validation.',
)
parser.add_argument(
    '-out_dir',
    type=str,
    help='Directory to use for saving trained model and validation results.',
)
args = parser.parse_args()

fp_config = f'{args.config_dir}/config.json'
with open(fp_config) as f:
    config = json.load(f)

out_dir = args.out_dir or f'{args.config_dir}/outputs/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

nsrdb_files = None if not args.nsrdb_fps else glob(args.nsrdb_fps)
cache_pattern = args.cache_fps or os.path.join(out_dir, 'mlclouds_df_{}.csv')
files = [
    args.surfrad_fps.format(year=y, area=ew)
    for y in args.years
    for ew in ('east', 'west')
]
files_e = [
    args.surfrad_fps.format(year=y, area=ew)
    for y in args.years
    for ew in ('east',)
]
files_w = [
    args.surfrad_fps.format(year=y, area=ew)
    for y in args.years
    for ew in ('west',)
]

logger.info('Number of files: %s', len(files))
logger.info('Number of east files: %s', len(files_e))
logger.info('Number of west files: %s', len(files_w))
logger.info('Source files: %s', files)
logger.info('Full config: %s', config)


fp_history = os.path.join(out_dir, 'training_history.csv')
fp_model = os.path.join(out_dir, 'mlclouds_model.pkl')
fp_env = os.path.join(out_dir, 'mlclouds_model_env.json')
fp_stats = os.path.join(out_dir, 'validation_stats.csv')
fp_stats_e = os.path.join(out_dir, 'validation_stats_east.csv')
fp_stats_w = os.path.join(out_dir, 'validation_stats_west.csv')
file_iter = (files, files_e, files_w)
fp_iter = (fp_stats, fp_stats_e, fp_stats_w)


if __name__ == '__main__':
    t = Trainer(
        train_sites=args.sites,
        train_files=files,
        config=config,
        test_fraction=0.2,
        nsrdb_files=nsrdb_files,
        cache_pattern=cache_pattern,
    )

    t.model.history.to_csv(fp_history)
    t.model.save_model(fp_model)
    with open(fp_env, 'w') as f:
        json.dump(t.model.version_record, f)

    for val_files, fp_stats_out in zip(file_iter, fp_iter):
        file_mask = np.isin(t.train_data.observation_sources, val_files)
        test_set_mask = t.test_set_mask.copy()[file_mask]
        logger.info(
            'Validating on %s out of %s observations',
            test_set_mask.sum(),
            len(test_set_mask),
        )
        v = Validator(
            t.model,
            config=config,
            val_sites=args.sites,
            val_files=val_files,
            save_timeseries=False,
            test_set_mask=test_set_mask,
        )
        v.stats.to_csv(fp_stats_out)
