"""Misc utilities for mlclouds training and validation."""

import datetime
import json
import logging
import os
import re

import pandas as pd

from mlclouds import CONFIG_FPATH
from mlclouds.p_fun import p_fun_all_sky, p_fun_dummy

logger = logging.getLogger(__name__)

ALL_SKY_VARS = (
    'alpha',
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
    'total_precipitable_water',
)

# Baseline nsrdb irradiance for stats
#   year - four digit year, e.g. 2017
#   yy - two digit eyar, e.g. 17
#   area - east or west
FP_BASELINE = (
    '/projects/pxs/mlclouds/training_data/{year}_{area}_v322/final/*.h5'
)

# Default satellite data for model training and validation
FP_DATA = (
    '/projects/pxs/mlclouds/training_data/{year}_{area}_v322/'
    'mlclouds_surfrad_{area}_{year}.h5'
)

# Ground measurement data
FP_SURFRAD_DATA = '/projects/pxs/surfrad/h5/{code}_{year}.h5'
FP_SURFRAD_META = '/projects/pxs/reference_grids/surfrad_meta.csv'

# Training data prep options for clean_cloud_df()
TRAINING_PREP_KWARGS = {
    'filter_daylight': True,
    'filter_clear': False,
    'filter_sky_class': False,
    'add_cloud_flag': True,
    'sza_lim': 89,
    'nan_option': 'interp',
}

P_FUNS = {'p_fun_all_sky': p_fun_all_sky, 'p_fun_dummy': p_fun_dummy}


# Phygnn model configuration
with open(CONFIG_FPATH) as f:
    CONFIG = json.load(f)


def get_valid_surf_sites(sites, fp_surfrad_data, data_file):
    """Get surfrad sites available for the given data file. This is
    determined from the year in the data file name.

    Parameters
    ----------
    sites: list
        List of surfrad gids to check. Considered valid if there exist files
        for these sites.
    fp_surfrad_data: str
        File path pattern for surfrad data files. Must include {year} and
        {code} format keys, where "code" is the string identifier for a given
        surfrad site.

    Returns
    -------
    valid_sites: list
        List of surfrad site gids with existing data files.

    """
    year, _ = extract_file_meta(data_file)
    valid_sites = []
    for gid in sites:
        surfrad_file = fp_surfrad_data.format(
            year=year, code=surf_meta().loc[gid, 'surfrad_id']
        )
        if os.path.exists(surfrad_file):
            valid_sites.append(gid)
    return valid_sites


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
    year = int(match_year.group())

    search_area = re.compile('(east|west)')
    match_area = search_area.search(fname)
    if match_area is None:
        msg = 'No area found in filename {}'.format(fname)
        logger.critical(msg)
        raise ValueError(msg)
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
        msg = (
            'Calculated time step of {} minutes is not an integer. This is '
            'likely an error'.format(minutes)
        )
        logger.error(msg)
        raise ValueError(msg)

    return int(minutes)


def surf_meta():
    """Return dataframe of surfrad gids and three letter codes"""
    surf_meta = pd.read_csv(FP_SURFRAD_META, index_col=0)
    surf_meta.index.name = 'gid'
    return surf_meta[['surfrad_id']]
