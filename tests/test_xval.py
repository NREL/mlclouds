"""
These are tests for the cross validation class. Also contains functions to
create the statistics csv files that are used in the tests. As a general rule,
the create_*() functions should NOT be run unless there is a modification to
PhyGNN, tensor_flow, or the mlclouds packages that justifies it.
"""
import os
import pytest
import tempfile
import pytz
from datetime import datetime as dt

from mlclouds.autoxval import XVal
from mlclouds.utilities import CONFIG, extract_file_meta, FP_DATA

STAT_FIELDS = ['MAE (%)', 'MBE (%)', 'RMSE (%)']


def check_for_eagle():
    if not os.path.exists('/lustre/eaglefs/projects/pxs/mlclouds/'):
        msg = ('These tests require access to /projects/pxs/mlclouds/ and '
               'can only be run on the Eagle HPC')
        pytest.skip(msg)


def test_xval():
    """
    Test that xval creates the proper results for a simple model. Also test
    model saving and loading.
    """
    check_for_eagle()

    config = CONFIG
    config['epochs_a'] = 4
    config['epochs_b'] = 4

    xv = XVal(config=config)
    fp_data = FP_DATA.format(year=2018, area='east')
    xv.train(train_sites=[0], train_files=fp_data)

    good_shape = (105120, len(CONFIG['features']) + 4)
    assert xv._train_data.df_raw.shape == good_shape

    # Unable to save stats before validating
    with pytest.raises(RuntimeError):
        xv.save_stats('exception.csv')

    xv2 = XVal()
    with tempfile.TemporaryDirectory() as td:
        xv.save_model(os.path.join(td, 'model.pkl'))
        xv2.load_model(os.path.join(td, 'model.pkl'))
    assert xv2._config['epochs_a'] == 4

    xv2.validate(val_files=fp_data)

    mask1 = xv2.stats['Variable'] == 'GHI'
    mask2 = xv2.stats['Condition'] == 'All-Sky'
    mask3 = xv2.stats['Model'] == 'MLClouds'
    mask4 = xv2.stats['Model'] == 'Baseline'
    base_mask = mask1 & mask2 & mask4
    ml_mask = mask1 & mask2 & mask3

    assert (xv2.stats.loc[base_mask, 'MAE (%)'] < 30).all()
    assert (xv2.stats.loc[ml_mask, 'MAE (%)'] < 30).all()


def test_xval_mismatched_timesteps():
    """
    Test training and validation with 5 minute and 30 minute GOES data at
    the same time.
    """
    check_for_eagle()

    config = CONFIG
    config['epochs_a'] = 4
    config['epochs_b'] = 4

    files = [FP_DATA.format(year=2018, area='west'),
             FP_DATA.format(year=2019, area='west')]

    xv = XVal(config=config)
    xv.train(train_sites=[4, 5], train_files=files)

    # correct amount of data for 2 sites with 1 year @ 30 min and 1 @ 10 min
    good_shape = ((2*24*365 + 6*24*365)*2, len(CONFIG['features']) + 4)
    assert xv._train_data.df_raw.shape == good_shape

    date = dt(2019, 1, 1, 0, 0, 0, 0, pytz.UTC)
    assert (xv._train_data.df_raw.time_index >= date).sum() == 6*24*365*2
    assert (xv._train_data.df_raw.time_index < date).sum() == 2*24*365*2

    xv.validate(val_files=files)

    mask1 = xv.stats['Variable'] == 'GHI'
    mask2 = xv.stats['Condition'] == 'All-Sky'
    mask3 = xv.stats['Model'] == 'MLClouds'
    mask4 = xv.stats['Model'] == 'Baseline'
    base_mask = mask1 & mask2 & mask4
    ml_mask = mask1 & mask2 & mask3

    assert (xv.stats.loc[base_mask, 'MAE (%)'].mean() < 30)
    assert (xv.stats.loc[ml_mask, 'MAE (%)'].mean() < 40)


def test_extract_file_meta():
    """ Test year and area extraction from file names """
    files = [FP_DATA.format(year=2018, area='east'),
             FP_DATA.format(year=2019, area='east'),
             FP_DATA.format(year=2018, area='west'),
             FP_DATA.format(year=2019, area='west'),
             ]
    years = [(2018, 'east'), (2019, 'east'), (2018, 'west'), (2019, 'west')]
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


if __name__ == '__main__':
    execute_pytest()
