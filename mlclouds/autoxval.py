"""
Automatic cross validation of PHYGNN models predicting opd and reff

Mike Bannister 7/2020
Based on code by Grant Buster
"""
import pandas as pd
import numpy as np
import os
import logging
import plotly.express as px
import json
from copy import deepcopy

from mlclouds.data_handlers import ValidationData
from mlclouds.trainer import Trainer
from mlclouds.validator import Validator
from mlclouds.utilities import FP_DATA, ALL_SKY_VARS, CONFIG, surf_meta

from phygnn import PhygnnModel as Phygnn

logger = logging.getLogger(__name__)


class TrainTest:
    """
    Train and test a model using a single dataset. Reserve a percentage
    of the data for testing that is not used for training.
    """

    def __init__(self, data_files, config=CONFIG, test_fraction=0.2,
                 stats_file=None, model_file=None, history_file=None):
        """
        Parameters
        ----------
        data_files: list of str | str
            File or list of files to use for training and testing. Filenames
            must include the four-digit year and satellite indicator
            (east|west).
        config: dict
            Phygnn configuration dict
        test_fraction: float
            Fraction of full data set to reserve for testing. Should be between
            0 to 1. The test set is randomly selected and dropped from the
            training set.
        stats_file: str | None
            If str, save stats to stats_file
        model_file: str | None
            If str, save model to model_file (pkl and json)
        history_file : str | None
            If str, save model training history to history_file.
        """
        self.trainer = Trainer(train_files=data_files, config=config,
                               test_fraction=test_fraction)
        self.validator = Validator(self.trainer.model, config=config,
                                   val_files=data_files,
                                   test_set_mask=self.trainer.test_set_mask)

        self._config = config
        self._model = self.trainer.model

        if stats_file:
            self.validator.stats.to_csv(stats_file)

        if model_file:
            self.save_model(model_file)

        if history_file:
            self.save_history(history_file)

    def save_model(self, fp):
        """
        Save model to disk

        Parameters
        ----------
        fp: str
            File name and path for model file (pkl and json)
        """
        self._model.save_model(fp)

    def save_history(self, fp):
        """
        Save model training history to disk

        Parameters
        ----------
        fp: str
            File name and path for csv
        """
        self._model.history.to_csv(fp)


class XVal:
    """
    Train a PHYGNN using one or more satellite datasets then validate against
    the NSRDB baseline data using another satellite dataset to predict cloud
    parameters. The sites used for training may also be controlled.
    """

    def __init__(self, config=CONFIG):
        """
        Parameters
        ----------
        config: dict
            Dict of configuration options. See CONFIG for example.
        """
        self._config = config
        self._model = None
        self._train_data = None
        self.stats = None
        self._validator = None

    def train(self, train_sites=[0, 1, 2, 3, 5, 6], train_files=FP_DATA):
        """
        Train PHYGNN model

        Parameters
        ----------
        train_sites: list of int
            Sites to use for training
        train_files: list | str
            File or list of files to use for training. Filenames must include
            the four-digit year and satellite domain (east|west).
        """
        trainer = Trainer(train_sites=train_sites, train_files=train_files,
                          config=self._config)

        self._model = trainer.model
        self._train_data = trainer.train_data
        self._config['train_files'] = self._train_data.train_files
        self._config['train_sites'] = self._train_data.train_sites

    def validate(self, val_files=None, val_data=None, update_clear=False,
                 update_cloudy=False, save_timeseries=False):
        """
        Predict values using PHYGNN model and validation against baseline
        NSRDB data.

        val_files: str | list of str | None
            File or list of file to use for validation from config file.
            Filenames must include the four-digit year and east/west text to
            indicate satellite. Must be None if val_data is set.
        val_data: None | ValidationData instance
            Use preloaded validation data or load from val_files if None. Must
            be none if val_files is set.
        update_clear: bool
            If true, update cloud type for clear time steps with phygnn
            predictions
        update_cloudy: bool
            If true, update cloud type for cloudy time steps with phygnn
            predictions
        save_timeseries: bool
            Save time series data to disk
        """
        if self._model is None or self._config is None:
            msg = 'A model must be trained or loaded before validating.'
            logger.critical(msg)
            raise RuntimeError(msg)

        validator = Validator(self._model,
                              config=self._config,
                              val_files=val_files,
                              val_data=val_data,
                              update_clear=update_clear,
                              update_cloudy=update_cloudy,
                              save_timeseries=save_timeseries)
        self.stats = validator.stats
        self._validator = validator

    def load_model(self, fname):
        """
        Load existing model from disk

        Parameters
        ----------
        fname: str
            File name and path of pickle file
        """
        self._model = Phygnn.load(fname)
        with open(fname + '.config', 'rb') as f:
            self._config = json.load(f)

    def save_model(self, fname):
        """
        Save model to disk

        Parameters
        ----------
        fname: str
            File name and path for pickle file
        """
        self._model.save_model(fname)
        with open(fname + '.config', 'w') as f:
            json.dump(self._config, f)

    def save_stats(self, fname):
        """
        Save statistics and model config to file

        Parameters
        ----------
        fname: str
            File name and path for stats CSV file
        """
        if self.stats is None:
            msg = 'Statistics do not exist. Run XVal.validate() first'
            logger.critical(msg)
            raise RuntimeError(msg)

        self.stats.to_csv(fname, index=False)
        conf = deepcopy(self._config)

        conf['val_files'] = self._validator.val_data.val_files
        conf['val_files_meta'] = self._validator.val_data.files_meta
        with open(fname[:-4] + '.json', 'wt') as f:
            json.dump(conf, f, indent=4)

    def plot(self, gid):
        """
        Show statistics bar charts

        Parameters
        ----------
        gid: int
            gid code of desired surfrad site to plot statistics for.
        """
        code = surf_meta().loc[gid, 'surfrad_id']
        for ylabel in ['MBE (%)', 'MAE (%)', 'RMSE (%)']:
            fig = px.bar(self.stats[(self.stats.Site == code.upper())],
                         x="Condition", y=ylabel, color='Model',
                         facet_col="Variable", barmode='group', height=400)
            fig.show()


class AutoXVal:
    """
    Run cross validation by both varying the number of sites used for
    training, and the site used for validation.
    """
    def __init__(self, sites=[0, 1, 2, 3, 4, 5, 6], val_sites=None,
                 data_files=FP_DATA, val_data=None, config=CONFIG,
                 shuffle_train=False, seed=None, xval=XVal, catch_nan=False,
                 min_train=1, save_timeseries=False):
        """
        Parameters
        ----------
        sites: list
            Sites to use for training and validation
        val_sites: None | int | list
            Site(s) to use for validation, use all if None
        data_files: str | list
            Files to use for training and validation
        val_data: ValidationData instance | None
            Validation data to use. Load from data_files if None.
        config: dict
            Dict of XVal configuration options. See CONFIG for example.
        shuffle_train: bool
            Randomize training site list before iterating over # of training
            sites.
        seed: None | int
            Seed for numpy.random if int
        xval: Class
            Cross validation class. Used for testing.
        catch_nan: bool
            If true, catch loss==nan exceptions and continue analysis
        min_train: int
            Minimum # of sites to use for training
        save_timeseries: bool
            Save time series data to disk
        """
        if seed is not None:
            np.random.seed(seed)

        if val_sites is None:
            val_sites = sites
        elif isinstance(val_sites, int):
            val_sites = [val_sites]
        elif isinstance(val_sites, str):
            val_sites = [int(val_sites)]

        self._sites = sites
        self._val_sites = val_sites
        self._config = config
        self._data_files = data_files

        logger.info('AXV: training sites are {}, val sites are {}'
                    ''.format(sites, val_sites))

        self.temp_stats = None

        if val_data is None:
            logger.info('Loading validation data from {}'.format(data_files))
            val_data = ValidationData(data_files, features=config['features'],
                                      y_labels=config['y_labels'],
                                      all_sky_vars=ALL_SKY_VARS,
                                      one_hot_cats=config['one_hot_categories']
                                      )
        self._val_data = val_data

        for val_site in val_sites:
            train_set = [x for x in sites if x != val_site]
            if shuffle_train:
                np.random.shuffle(train_set)
            self._run_train_set(val_site, train_set, min_train, catch_nan,
                                xval, save_timeseries=save_timeseries)

    def _run_train_set(self, val_site, train_set, min_train, catch_nan, xval,
                       save_timeseries=False):
        """
        Run XVal with permutations of a training set. "min_train" controls the
        number of permutations.

        Parameters
        ----------
        val_site: int
            Site to use for validation
        train_set: list
            Set of sites to use for training
        catch_nan: bool
            If true, catch loss==nan exceptions and continue analysis
        min_train: int
            Minimum # of sites to use for training
        xval: Class
            Cross validation class. Used for testing.
        save_timeseries: bool
            Save time series data to disk
        """
        logger.info('Training set is {}, val site is {}'.format(train_set,
                                                                val_site))
        for i in range(min_train - 1, len(train_set)):
            train_sites = train_set[0:i + 1]
            xv = xval(config=self._config)
            try:
                xv.train(train_sites=train_sites, train_files=self._data_files)
            except ArithmeticError as e:
                if catch_nan:
                    logger.warning('Loss=nan, val on {}, train on {}'
                                   ''.format(val_site, train_sites))
                    continue
                else:
                    raise e
            xv.validate(val_data=self._val_data,
                        save_timeseries=save_timeseries)

            # The _ prevents the 0 from being trimmed off
            ts = '_' + ''.join([str(x) for x in train_sites])
            xv.stats['val_site'] = val_site
            xv.stats['train_sites'] = ts
            xv.stats['num_ts'] = len(ts) - 1

            if self.temp_stats is None:
                self.temp_stats = xv.stats
            else:
                self.temp_stats = pd.concat([self.temp_stats, xv.stats])

        self.stats = self.temp_stats.reset_index()

    def save_stats(self, path='./stats', fname=None):
        """
        Save validation statistics csv and config as json

        Parameters
        ----------
        path: str
            Path to save stats csv and config json to
        fname: str | None
            Filename w/o extension for stats and config files. Auto generate if
            None.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        if fname is None:
            sites_name = ''.join([str(x) for x in self._sites])
            val_name = ''.join([str(x) for x in self._val_sites])
            fname = 'axv_stats_{}_{}'.format(sites_name, val_name)

        fpath = os.path.join(path, fname)
        self.stats.to_csv(fpath + '.csv')
        logger.info('Saved stats to: {}'.format(fpath + '.csv'))

        conf = deepcopy(self._config)
        conf['data_files'] = self._data_files
        conf['val_files'] = self._val_data.val_files
        conf['sites'] = self._sites
        conf['val_sites'] = self._val_sites
        with open(fpath + '.json', 'wt') as f:
            json.dump(conf, f, indent=4)
        logger.info('Saved config to: {}'.format(fpath + '.json'))

    @classmethod
    def k_fold(cls, data_files=FP_DATA, val_data=None,
               sites=[0, 1, 2, 3, 4, 5, 6], val_sites=None, config=CONFIG,
               seed=None, xval=XVal, catch_nan=False, save_timeseries=False):
        """ Perform k-fold validation, only train on n-1 sites """
        min_train = len(sites) - 1
        axv = cls(data_files=data_files, val_data=val_data, sites=sites,
                  val_sites=val_sites, config=config, seed=seed, xval=xval,
                  catch_nan=catch_nan, min_train=min_train,
                  save_timeseries=save_timeseries)
        return axv

    @classmethod
    def kxn_fold(cls, data_files=FP_DATA, val_data=None,
                 sites=[0, 1, 2, 3, 4, 5, 6], val_sites=None, config=CONFIG,
                 shuffle_train=False, seed=None, xval=XVal, catch_nan=False,
                 min_train=1, save_timeseries=False):
        """ Perform cross validation against subsets of training sites """
        axv = cls(data_files=data_files, val_data=val_data, sites=sites,
                  val_sites=val_sites, config=config,
                  shuffle_train=shuffle_train, seed=seed, xval=xval,
                  catch_nan=catch_nan, min_train=min_train,
                  save_timeseries=save_timeseries)
        return axv
