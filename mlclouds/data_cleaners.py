"""Training data cleaning methods"""

import logging
import time

import numpy as np
import pandas as pd
from farms import CLEAR_TYPES, ICE_TYPES, WATER_TYPES

logger = logging.getLogger(__name__)


def sky_class_filter(cloud_df):
    """Filter cloud data so that final data includes only data with matching
    sky_class and cloud types. ``sky_class`` is determined by comparing clear
    sky REST2 predictions with ground measurements. If the clear sky prediction
    is within a given tolerance the class is considered "clear", otherwise the
    class is considered "cloudy"."""

    cloudy = cloud_df['cloud_type'].isin(ICE_TYPES + WATER_TYPES)
    clear = cloud_df['cloud_type'].isin(CLEAR_TYPES)

    if 'sky_class' in cloud_df.columns:
        cloudy &= cloud_df['sky_class'].isin(('cloudy',))
        clear &= cloud_df['sky_class'].isin(('clear',))

    return clear | cloudy


def clean_cloud_df(
    cloud_df_raw,
    filter_daylight=True,
    filter_clear=False,
    add_cloud_flag=True,
    sza_lim=89,
    nan_option='interp',
):
    """Clean up cloud data. Includes options to filter night and clear
    timesteps, timesteps with sza which exceed a given threshold, and to add
    cloud flag labels"""
    t0 = time.time()
    cloud_df = cloud_df_raw.copy()
    day = cloud_df['solar_zenith_angle'] < sza_lim

    day_missing_ctype = day & (cloud_df['cloud_type'] < 0)
    cloud_df.loc[(cloud_df['cloud_type'] < 0), 'cloud_type'] = np.nan
    cloud_df['cloud_type'] = (
        cloud_df['cloud_type'].interpolate('nearest').ffill().bfill()
    )

    cloudy = cloud_df['cloud_type'].isin(ICE_TYPES + WATER_TYPES)
    day_clouds = day & cloudy
    day_missing_opd = day_clouds & (cloud_df['cld_opd_dcomp'] <= 0)
    day_missing_reff = day_clouds & (cloud_df['cld_reff_dcomp'] <= 0)
    cloud_df.loc[(cloud_df['cld_opd_dcomp'] <= 0), 'cld_opd_dcomp'] = np.nan
    cloud_df.loc[(cloud_df['cld_reff_dcomp'] <= 0), 'cld_reff_dcomp'] = np.nan

    logger.info(
        '{:.2f}% of timesteps are daylight'.format(100 * day.sum() / len(day))
    )
    logger.info(
        '{:.2f}% of daylight timesteps are cloudy'.format(
            100 * day_clouds.sum() / day.sum()
        )
    )
    logger.info(
        '{:.2f}% of daylight timesteps are missing cloud type'.format(
            100 * day_missing_ctype.sum() / day.sum()
        )
    )
    logger.info(
        '{:.2f}% of cloudy daylight timesteps are missing cloud opd'.format(
            100 * day_missing_opd.sum() / day_clouds.sum()
        )
    )
    logger.info(
        '{:.2f}% of cloudy daylight timesteps are missing cloud reff'.format(
            100 * day_missing_reff.sum() / day_clouds.sum()
        )
    )

    logger.debug('Column NaN values:')
    for c in cloud_df.columns:
        pnan = 100 * pd.isna(cloud_df[c]).sum() / len(cloud_df)
        logger.debug('\t"{}" has {:.2f}% NaN values'.format(c, pnan))

    if 'interp' in nan_option.lower():
        logger.debug('Interpolating opd and reff')

        if 'time_index' in cloud_df.columns:
            time_index = cloud_df.time_index
            assert time_index.isnull().sum() == 0
            cloud_df = cloud_df.drop('time_index', axis=1)
            cloud_df = cloud_df.interpolate('nearest').ffill().bfill()
            cloud_df['time_index'] = time_index
        else:
            cloud_df = cloud_df.interpolate('nearest').ffill().bfill()

        cloud_df.loc[~cloudy, 'cld_opd_dcomp'] = 0.0
        cloud_df.loc[~cloudy, 'cld_reff_dcomp'] = 0.0
    elif 'drop' in nan_option.lower():
        l0 = len(cloud_df)
        cloud_df = cloud_df.dropna(axis=0, how='any')
        day = cloud_df['solar_zenith_angle'] < sza_lim
        cloudy = cloud_df['cloud_type'].isin(ICE_TYPES + WATER_TYPES)
        logger.debug(
            'Dropped {} rows with NaN values.'.format(l0 - len(cloud_df))
        )

    assert ~any(cloud_df['cloud_type'] < 0)
    assert ~any(pd.isna(cloud_df))
    assert ~any(cloudy & (cloud_df['cld_opd_dcomp'] <= 0))

    if add_cloud_flag:
        logger.debug(
            'Adding cloud type flag (e.g. flag=[night, clear, '
            'ice_cloud, water_cloud, bad_cloud])'
        )
        ice_clouds = cloud_df['cloud_type'].isin(ICE_TYPES)
        water_clouds = cloud_df['cloud_type'].isin(WATER_TYPES)
        cloud_df['flag'] = 'night'
        cloud_df.loc[day, 'flag'] = 'clear'
        cloud_df.loc[ice_clouds, 'flag'] = 'ice_cloud'
        cloud_df.loc[water_clouds, 'flag'] = 'water_cloud'
        cloud_df.loc[day_missing_ctype, 'flag'] = 'bad_cloud'
        cloud_df.loc[day_missing_opd, 'flag'] = 'bad_cloud'
        cloud_df.loc[day_missing_reff, 'flag'] = 'bad_cloud'

    mask = True
    if filter_daylight:
        mask &= day

    if filter_clear:
        mask &= cloudy

    if filter_clear or filter_daylight:
        logger.info(
            'Data reduced from '
            '{} rows to {} after filters ({:.2f}% of original)'.format(
                len(cloud_df), mask.sum(), 100 * mask.sum() / len(cloud_df)
            )
        )

        cloud_df = cloud_df[mask]

    if add_cloud_flag:
        logger.debug(
            'Feature flag column has these values: {}'.format(
                cloud_df.flag.unique()
            )
        )
    logger.info('Cleaning took {:.1f} seconds'.format(time.time() - t0))

    return cloud_df
