"""
mlclouds phygnn model trainer and validator classes.
"""

import logging

from phygnn import PhygnnModel as Phygnn

from mlclouds.data_handlers import TrainData
from mlclouds.utilities import FP_DATA, P_FUNS, CONFIG, surf_meta

logger = logging.getLogger(__name__)


class Trainer:
    """Class to handle the training of the mlclouds phygnn model"""

    def __init__(self, train_sites='all', train_files=FP_DATA, config=CONFIG,
                 test_fraction=None, fp_save_data=None):
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
        fp_save_data : str
            Optional .csv filepath to save training data to
        """

        logger.info('Trainer: Training on sites {} from files {}'
                    ''.format(train_sites, train_files))
        if train_sites == 'all':
            train_sites = [k for k, v in
                           surf_meta().to_dict()['surfrad_id'].items()]
        self.train_sites = train_sites
        self.train_files = train_files
        self._config = config

        logger.info('Trainer: Training on sites {} from files {}'
                    ''.format(train_sites, train_files))

        self.train_data = TrainData(self.train_sites, self.train_files,
                                    config=self._config,
                                    test_fraction=test_fraction)
        self.x = self.train_data.x
        self.y = self.train_data.y
        self.p = self.train_data.p
        self.test_set_mask = self.train_data.test_set_mask

        if fp_save_data is not None:
            self.train_data.save_all_data(fp_save_data)

        self.p_kwargs = {'labels':
                         self.train_data.df_all_sky.columns.values.tolist()}
        self.p_kwargs.update(self._config.get('p_kwargs', {}))

        logger.debug('Building PHYGNN model')

        p_fun = P_FUNS[self._config.get('p_fun', 'p_fun_all_sky')]
        logger.info('Using p_fun: {}'.format(p_fun))

        Phygnn.seed(self._config['phygnn_seed'])
        one_hot_categories = self._config['one_hot_categories']

        model = Phygnn.build(p_fun=p_fun,
                             one_hot_categories=one_hot_categories,
                             feature_names=list(self.x.columns),
                             label_names=self._config['y_labels'],
                             hidden_layers=self._config['hidden_layers'],
                             loss_weights=self._config['loss_weights_a'],
                             metric=self._config['metric'],
                             learning_rate=self._config['learning_rate'])

        logger.info('Training part A - pure data. Loss is {}'
                    ''.format(self._config['loss_weights_a']))

        out = model.train_model(self.x, self.y, self.p,
                                n_batch=self._config['n_batch'],
                                n_epoch=self._config['epochs_a'],
                                p_kwargs=self.p_kwargs)

        logger.info('Training part B - data and phygnn. Loss is {}'
                    ''.format(self._config['loss_weights_b']))
        model.set_loss_weights(self._config['loss_weights_b'])
        out = model.train_model(self.x, self.y, self.p,
                                n_batch=self._config['n_batch'],
                                n_epoch=self._config['epochs_b'],
                                p_kwargs=self.p_kwargs)

        self.model = model
        self.out = out
        logger.info('Training complete')
