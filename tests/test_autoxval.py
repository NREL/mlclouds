import os
import pandas as pd
import pytest

from mlclouds.autoxval import AutoXVal
from mlclouds import TESTDATADIR


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

    def validate(self, val_data=None):
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
