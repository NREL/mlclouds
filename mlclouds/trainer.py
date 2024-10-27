"""MLClouds phygnn model trainer and validator classes."""

import logging

from mlclouds.data_handlers import TrainData
from mlclouds.model.base import MLCloudsModel
from mlclouds.utilities import CONFIG, FP_DATA, P_FUNS, surf_meta

logger = logging.getLogger(__name__)


class Trainer:
    """Class to handle the training of the mlclouds phygnn model"""

    def __init__(
        self,
        train_sites='all',
        train_files=FP_DATA,
        config=CONFIG,
        test_fraction=None,
        nsrdb_files=None,
        cache_pattern=None,
    ):
        """
        Train PHYGNN model

        Parameters
        ----------
        train_sites: 'all' | list of int
            Surfrad gids to use for training. Use all if 'all'
        train_files: list of str | str
            File or list of files to use for training. Filenames must include
            the four-digit year and satellite indicator (east|west).
        config: dict
            Phygnn configuration dict
        test_fraction: None | float
            Fraction of full data set to reserve for testing. Should be between
            0 to 1. The test set is randomly selected and dropped from the
            training set. If None, do not reserve a test set.
        nsrdb_files : list | str
            Nsrdb files including irradiance data for the training sites. This
            is used to compute the sky class for these locations which is then
            used to filter cloud type data for false positives / negatives
        cache_pattern : str
            Optional .csv filepath pattern to save data to. e.g.
            ``./df_{}.csv``. This will be used to save
            ``self.train_data.df_raw`` and ``self.train_data.df_all_sky``
            before they have been split into training and validation sets
        train_kwargs : dict | None
            Dictionary of keyword args for ``model.train_model``. e.g.
            ``run_preflight, return_diagnostics, etc``
        """

        logger.info(
            'Trainer: Training on sites {} from files {}' ''.format(
                train_sites, train_files
            )
        )
        if train_sites == 'all':
            train_sites = [
                k for k, v in surf_meta().to_dict()['surfrad_id'].items()
            ]
        self.train_sites = train_sites
        self.train_files = train_files
        self._config = config

        logger.info(
            'Trainer: Training on sites {} from files {}' ''.format(
                train_sites, train_files
            )
        )

        self.train_data = TrainData(
            train_sites=self.train_sites,
            train_files=self.train_files,
            config=self._config,
            test_fraction=test_fraction,
            nsrdb_files=nsrdb_files,
            cache_pattern=cache_pattern,
        )

        self.x = self.train_data.x
        self.y = self.train_data.y
        self.p = self.train_data.p
        self.test_set_mask = self.train_data.test_set_mask

        self.p_kwargs = {
            'labels': self.train_data.df_all_sky.columns.values.tolist()
        }
        self.p_kwargs.update(self._config.get('p_kwargs', {}))

        logger.debug('Building PHYGNN model')

        p_fun = P_FUNS[self._config.get('p_fun', 'p_fun_all_sky')]
        logger.info('Using p_fun: {}'.format(p_fun))

        MLCloudsModel.seed(self._config['phygnn_seed'])
        one_hot_categories = self._config['one_hot_categories']

        model = MLCloudsModel.build(
            p_fun=p_fun,
            one_hot_categories=one_hot_categories,
            feature_names=list(self.x.columns),
            label_names=self._config['y_labels'],
            hidden_layers=self._config['hidden_layers'],
            loss_weights=self._config['loss_weights_a'],
            metric=self._config['metric'],
            learning_rate=self._config['learning_rate'],
        )
        logger.info(
            'Training part A - pure data. Loss is {}' ''.format(
                self._config['loss_weights_a']
            )
        )

        out = model.train_model(
            self.x,
            self.y,
            self.p,
            n_batch=self._config['n_batch'],
            n_epoch=self._config['epochs_a'],
            run_preflight=self._config.get('run_preflight', True),
            p_kwargs=self.p_kwargs,
        )
        logger.info(
            'Training part B - data and phygnn. Loss is {}' ''.format(
                self._config['loss_weights_b']
            )
        )
        model.set_loss_weights(self._config['loss_weights_b'])
        out = model.train_model(
            self.x,
            self.y,
            self.p,
            n_batch=self._config['n_batch'],
            n_epoch=self._config['epochs_b'],
            run_preflight=self._config.get('run_preflight', True),
            p_kwargs=self.p_kwargs,
        )

        self.model = model
        self.out = out
        logger.info('Training complete')
