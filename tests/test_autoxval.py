import os
import pandas as pd
import pytest

from mlclouds.autoxval import AutoXVal
from mlclouds import TESTDATADIR
from mlclouds.utilities import FP_DATA, CONFIG, surf_meta
from mlclouds.data_handlers import TrainData


class FakeXVal:
    """ Verify that AutoXVal is sending the correct training sites """
    train_sets = [[1, 2, 3],
                  [0, 2, 3],
                  [0, 1, 3],
                  [0, 1, 2]]
    i = 0

    def __init__(self, config=0):
        stats_file = os.path.join(TESTDATADIR, 'fake_stats.csv')
        self.stats = pd.read_csv(stats_file)

    def train(self, train_sites=None, train_files=None):
        assert train_sites == self.train_sets[self.__class__.i]
        self.__class__.i += 1

    def validate(self, val_data=None, save_timeseries=False):
        pass


def test_kfold():
    """ Test kfold validation """
    FakeXVal.train_sets = [[1, 2, 3],
                           [0, 2, 3],
                           [0, 1, 3],
                           [0, 1, 2]]
    FakeXVal.i = 0

    axv = AutoXVal.k_fold(sites=[0, 1, 2, 3], val_data='fake',
                          xval=FakeXVal)
    assert len(axv.stats) == 4


def test_kxn_fold():
    """ Test kxn_fold validation """
    FakeXVal.train_sets = [[1],
                           [1, 2],
                           [0],
                           [0, 2],
                           [0],
                           [0, 1]]
    FakeXVal.i = 0

    axv = AutoXVal.kxn_fold(sites=[0, 1, 2], val_data='fake',
                            xval=FakeXVal)
    assert len(axv.stats) == 6


def test_test_train_split():
    """ Test train/test fraction has appropriate split """
    west_2016 = FP_DATA.format(year=2016, area='west')

    if not os.path.exists(west_2016):
        msg = ('These tests require access to /projects/pxs/mlclouds/ and '
               'can only be run on the Eagle HPC')
        pytest.skip(msg)

    td = TrainData('all', west_2016, config=CONFIG, test_fraction=0.2)
    n_surf = len(surf_meta())
    n_obs = 2 * 8784 * n_surf  # 30 min leap year data
    assert td.df_raw.shape[0] == int(round(n_obs * 0.8))
    assert td.test_set_mask.shape[0] == n_obs
    assert td.test_set_mask.sum() == int(round(n_obs * 0.2))

    # Verify the two masks are unique
    assert (td.test_set_mask | td.train_set_mask).sum() == n_obs
    assert (td.test_set_mask & td.train_set_mask).sum() == 0


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
