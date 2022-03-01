import re
import logging
import datetime
import pandas as pd

from mlclouds.p_fun import p_fun_all_sky, p_fun_dummy

logger = logging.getLogger(__name__)

ALL_SKY_VARS = ('alpha',
                'aod',
                'asymmetry',
                'cloud_type',
                'cld_opd_dcomp',
                'cld_reff_dcomp',
                'ozone',
                'solar_zenith_angle',
                'ssa',
                'surface_albedo',
                'surface_pressure',
                'total_precipitable_water')

# Baseline nsrdb irradiance for stats
#   year - four digit year, e.g. 2017
#   yy - two digit eyar, e.g. 17
#   area - east or west
FP_BASELINE = ('/projects/pxs/mlclouds/training_data/{year}_{area}_v311/'
               'final/*.h5')

# Default satellite data for model training and validation
FP_DATA = ('/projects/pxs/mlclouds/training_data/{year}_{area}_v321/'
           'mlclouds_surfrad_{area}_{year}.h5')

# Ground measurement data
FP_SURFRAD_DATA = '/projects/pxs/surfrad/h5/{code}_{year}.h5'
FP_SURFRAD_META = '/projects/pxs/reference_grids/surfrad_meta.csv'

# Training data prep options for clean_cloud_df()
TRAINING_PREP_KWARGS = {'filter_daylight': True,
                        'filter_clear': False,
                        'add_cloud_flag': True,
                        'sza_lim': 89,
                        'nan_option': 'interp'}

P_FUNS = {'p_fun_all_sky': p_fun_all_sky,
          'p_fun_dummy': p_fun_dummy}


# Phygnn model configuration
CONFIG = {
    'features': ['solar_zenith_angle', 'cloud_type',
                 'refl_0_65um_nom', 'refl_3_75um_nom',
                 'temp_3_75um_nom', 'cloud_transmission_0_65um_nom',
                 'cloud_fraction', 'air_temperature',
                 'dew_point', 'relative_humidity',
                 'total_precipitable_water', 'surface_albedo',
                 ],

    # Categories for one hot encoding. Keys are column names, values are lists
    # of category values
    'one_hot_categories': {'flag': ['clear', 'ice_cloud',
                                    'water_cloud', 'bad_cloud']},

    # Fields to predict using PhyGNN
    'y_labels': ['cld_opd_dcomp', 'cld_reff_dcomp'],

    # Neural network geometry
    'hidden_layers': [{'units': 64, 'activation': 'relu', 'name': 'relu1',
                       'dropout': 0.01},
                      {'units': 64, 'activation': 'relu', 'name': 'relu2',
                       'dropout': 0.01},
                      {'units': 64, 'activation': 'relu', 'name': 'relu3',
                       'dropout': 0.01},
                      ],

    # phygnn params
    'phygnn_seed': 0,
    'metric': 'relative_mae',
    'learning_rate': 0.01,
    'n_batch': 4,

    # Two training stages are used, with independent number of epochs and
    # loss weights. The loss weight tuple is (pure nn loss, physics loss). By
    # default the first stage of training ignores the physics loss and the
    # second stage weights the two losses evenly.
    'epochs_a': 100,
    'epochs_b': 100,
    'loss_weights_a': (1, 0),
    'loss_weights_b': (1, 1),
}


def extract_file_meta(fname):
    """
    Extract four-digit year and area from filename

    Parameters
    ----------
    fname: str
        Filename

    Returns
    -------
    year: int
        Year from filename
    area: str
        'east' or 'west'
    """
    search_year = re.compile('(20|19)[0-9][0-9]')

    match_year = search_year.search(fname)
    if match_year is None:
        msg = 'No four-digit year found in filename {}'.format(fname)
        logger.critical(msg)
        raise ValueError(msg)
    else:
        year = int(match_year.group())

    search_area = re.compile('(east|west)')
    match_area = search_area.search(fname)
    if match_area is None:
        msg = 'No area found in filename {}'.format(fname)
        logger.critical(msg)
        raise ValueError(msg)
    else:
        area = match_area.group()

    return year, area


def calc_time_step(series):
    """
    Extract time step from pandas time series. The mode is used to provide
    robustness against missing time steps.

    Parameters
    ----------
    series: pandas time series
        Time series to extract time step from.

    Returns
    -------
    minutes: int
        Size of time step in minutes.
    """
    mode = series.diff().mode().values[0]
    t_delta = mode.astype('timedelta64[m]').astype(datetime.datetime)
    minutes = t_delta.seconds / 60.0

    if not minutes.is_integer():
        msg = ('Calculated time step of {} minutes is not an integer. This is '
               'likely an error'.format(minutes))
        logger.error(msg)
        raise ValueError(msg)

    return int(minutes)


def surf_meta():
    """ Return data frame of surfrad gids and three letter codes """
    surf_meta = pd.read_csv(FP_SURFRAD_META, index_col=0)
    surf_meta.index.name = 'gid'
    return surf_meta[['surfrad_id']]
