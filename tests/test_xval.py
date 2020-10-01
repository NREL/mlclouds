"""
These are tests for the cross validation class. Also contains functions to
create the statistics csv files that are used in the tests. As a general rule,
the create_*() functions should NOT be run unless there is a modification to
PhyGNN, tensor_flow, or the mlclouds packages that justifies it.
"""
import os
import pandas as pd
import numpy as np
import pytest
import tempfile
import pytz
from datetime import datetime as dt

from mlclouds.autoxval import XVal
from mlclouds.utilities import CONFIG, extract_file_meta, FP_DATA
from mlclouds import TESTDATADIR

STAT_FIELDS = ['MAE (%)', 'MBE (%)', 'RMSE (%)']


@pytest.fixture
def check_for_eagle():
    if not os.path.exists('/lustre/eaglefs/projects/mlclouds/'):
        msg = 'These tests require access to /projects/mlclouds/ and can ' +\
              'only be run on the Eagle HPC'
        raise RuntimeError(msg)


def test_xval(check_for_eagle):
    """
    Test that xval creates the proper results for a simple model. Also test
    model saving and loading.
    """
    stats_file = os.path.join(TESTDATADIR, 'val_stats_0_6_2018_east_adj.csv')
    stats = pd.read_csv(stats_file)

    config = CONFIG
    config['epochs_a'] = 4
    config['epochs_b'] = 4

    xv = XVal(config=config)
    xv.train(train_sites=[0], train_files=FP_DATA)
    assert xv._train_data.df_raw.shape == (105120, 16)

    # Unable to save stats before validating
    with pytest.raises(RuntimeError):
        xv.save_stats('exception.csv')

    xv2 = XVal()
    with tempfile.TemporaryDirectory() as td:
        xv.save_model(os.path.join(td, 'model.pkl'))
        xv2.load_model(os.path.join(td, 'model.pkl'))
    assert xv2._config['epochs_a'] == 4

    xv2.validate(val_files=FP_DATA)

    for stat_field in STAT_FIELDS:
        print('testing', stat_field)
        assert np.isclose(stats[stat_field], xv2.stats[stat_field]).all()

    with tempfile.TemporaryDirectory() as td:
        csv_file = os.path.join(td, 'test.csv')
        json_file = os.path.join(td, 'test.json')
        xv2.save_stats(csv_file)
        assert os.path.isfile(csv_file)
        assert os.path.isfile(json_file)


def test_xval_two_year(check_for_eagle):
    """
    Test that xval creates the proper results for a simple model when
    training and validating on two years of data
    """
    stats_file = os.path.join(TESTDATADIR,
                              'val_stats_0_6_2018_2019_east_adj.csv')
    config = CONFIG
    config['epochs_a'] = 4
    config['epochs_b'] = 4

    files = ['/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2018' +
             '_east_adj/mlclouds_surfrad_2018_adj.h5',
             '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2019' +
             '_east_adj/mlclouds_surfrad_2019_adj.h5']

    xv = XVal(config=config)
    xv.train(train_sites=[0], train_files=files)
    assert xv._train_data.df_raw.shape == (210240, 16)

    xv.validate(val_files=files)

    stats = pd.read_csv(stats_file)

    for stat_field in STAT_FIELDS:
        print('testing', stat_field)
        assert np.isclose(stats[stat_field], xv.stats[stat_field]).all()


def test_xval_mismatched_timesteps(check_for_eagle):
    """
    Test training and validation with 5 minute and 30 minute GOES data at
    the same time.
    """
    stats_file = os.path.join(TESTDATADIR, 'val_stats_45_w_2018_2019.csv')
    stats = pd.read_csv(stats_file)

    config = CONFIG
    config['epochs_a'] = 4
    config['epochs_b'] = 4

    files = [
        # 2018 has a 30 minute timestep
        '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2018' +
        '_west_adj/mlclouds_surfrad_2018.h5',
        # 2019 has a 5 minute timestep
        '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2019' +
        '_west_adj/mlclouds_surfrad_2019.h5',
    ]

    xv = XVal(config=config)
    xv.train(train_sites=[4, 5], train_files=files)

    # Is there the correct amount of data for 1 year @ 30 min and 1 @ 5 min?
    assert xv._train_data.df_raw.shape == (245280, 16)
    date = dt(2019, 1, 1, 0, 0, 0, 0, pytz.UTC)
    assert (xv._train_data.df_raw.time_index >= date).sum() == 210240
    assert (xv._train_data.df_raw.time_index < date).sum() == 35040

    xv.validate(val_files=files)

    for stat_field in STAT_FIELDS:
        print('testing', stat_field)
        assert np.isclose(stats[stat_field], xv.stats[stat_field]).all()


def test_extract_file_meta():
    """ Test year and area extraction from file names """
    files = ['/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2018' +
             '_east_adj/mlclouds_surfrad_2018_adj.h5',
             '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/1918' +
             '_east_adj/mlclouds_surfrad_1918_adj.h5',
             '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2018' +
             '_west_adj/mlclouds_surfrad_2018_adj.h5',
             '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2017' +
             '_west_adj/mlclouds_surfrad_2017_adj.h5',
             ]
    years = [(2018, 'east'), (1918, 'east'), (2018, 'west'), (2017, 'west')]
    for meta, f in zip(years, files):
        y, a = extract_file_meta(f)
        assert y == meta[0]
        assert a == meta[1]

    with pytest.raises(ValueError):
        fname = 'no year or area'
        y, a = extract_file_meta(fname)

    with pytest.raises(ValueError):
        fname = 'east-bad year 190'
        y, a = extract_file_meta(fname)

    with pytest.raises(ValueError):
        fname = 'year (1999), bad area West'
        y, a = extract_file_meta(fname)


def execute_pytest(capture='all', flags='-rapP'):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


def create_test_xval_csv():
    """ Create csv stats file for test_xval() """
    stats_file = os.path.join(TESTDATADIR, 'val_stats_0_6_2018_east_adj.csv')

    config = CONFIG
    config['epochs_a'] = 4
    config['epochs_b'] = 4

    xv = XVal(config=config)
    xv.train(train_sites=[0], train_files=FP_DATA)
    xv.validate(val_files=FP_DATA)
    xv.save_stats(stats_file)


def create_test_xval_two_year_csv():
    """ Create csv stats file for test_xval_two_year() """
    stats_file = os.path.join(TESTDATADIR,
                              'val_stats_0_6_2018_2019_east_adj.csv')
    config = CONFIG
    config['epochs_a'] = 4
    config['epochs_b'] = 4

    files = ['/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2018' +
             '_east_adj/mlclouds_surfrad_2018_adj.h5',
             '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2019' +
             '_east_adj/mlclouds_surfrad_2019_adj.h5']

    xv = XVal(config=config)
    xv.train(train_sites=[0], train_files=files)
    xv.validate(val_files=files)
    xv.save_stats(stats_file)


def create_test_xval_mismatched_timesteps_csv():
    """ Create csv stats file for test_xval_mismatched_timesteps() """
    stats_file = os.path.join(TESTDATADIR, 'val_stats_45_w_2018_2019.csv')

    config = CONFIG
    config['epochs_a'] = 4
    config['epochs_b'] = 4

    files = [
        # 2018 has a 30 minute timestep
        '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2018' +
        '_west_adj/mlclouds_surfrad_2018.h5',
        # 2019 has a 5 minute timestep
        '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/2019' +
        '_west_adj/mlclouds_surfrad_2019.h5',
    ]

    xv = XVal(config=config)
    xv.train(train_sites=[4, 5], train_files=files)
    xv.validate(val_files=files)
    xv.save_stats(stats_file)


if __name__ == '__main__':
    execute_pytest()
