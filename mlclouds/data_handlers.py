import logging
import copy
import pandas as pd
import numpy as np

from rest2.rest2 import rest2, rest2_tuuclr

from farms.utilities import ti_to_radius, calc_beta
from farms import ICE_TYPES, WATER_TYPES

from nsrdb.file_handlers.surfrad import Surfrad

from mlclouds.data_cleaners import clean_cloud_df
from mlclouds.nsrdb import NSRDBFeatures
from mlclouds.utilities import ALL_SKY_VARS, CONFIG
from mlclouds.utilities import FP_SURFRAD_DATA, FP_SURFRAD_META
from mlclouds.utilities import TRAINING_PREP_KWARGS
from mlclouds.utilities import surf_meta, calc_time_step, extract_file_meta

logger = logging.getLogger(__name__)


class TrainData:
    """ Load and prep training data """
    def __init__(self, train_sites, train_files, config=CONFIG,
                 test_fraction=None):
        """
        Parameters
        ----------
        train_sites: 'all' | list of int
            Surfrad gids to use for training. Use all if 'all'
        train_files: list | str
            File or list of files to use for training. Filenames must include
            the four-digit year.
        config: dict
            Dict of configuration options. See CONFIG for example.
        test_fraction: None | float
            Fraction of full data set to reserve for testing. Should be between
            0 to 1. The test set is randomly selected and dropped from the
            training set. If None, do not reserve a test set.
        """

        self.fp_surfrad_data = FP_SURFRAD_DATA
        self.fp_surfrad_meta = FP_SURFRAD_META
        if train_sites == 'all':
            train_sites = [k for k, v in
                           surf_meta().to_dict()['surfrad_id'].items()]
        self.train_sites = train_sites
        self._config = config
        self.test_set_mask = None
        self.train_set_mask = None

        # keep a record of data sources with length equal to n observations
        self.observation_sources = []

        if not isinstance(train_files, list):
            train_files = [train_files]
        self.train_files = train_files

        logger.info('Loading training data')
        self._load_data(test_fraction)
        logger.info('Prepping training data')
        self._prep_data()

    def _load_data(self, test_fraction):
        """
        Load training data

        Parameters
        ----------
        test_fraction: None | float
            Fraction of full data set to reserve for testing. Should be between
            0 to 1. The test set is randomly selected and dropped from the
            training set. If None, do not reserve a test set.
        """
        var_names = copy.deepcopy(self._config['features'])
        var_names += self._config['y_labels']
        logger.debug('Loading vars {}'.format(var_names))

        df_raw = None
        df_all_sky = None
        df_surf = None
        for train_file in self.train_files:
            # ------ Grab NSRDB data for weather properties
            logger.debug('Loading data for site(s) {}, from {}'
                         .format(self.train_sites, train_file))
            with NSRDBFeatures(train_file) as res:
                temp_raw = res.extract_features(self.train_sites, var_names)
                temp_all_sky = res.extract_features(
                    self.train_sites,
                    self._config.get('all_sky_vars', ALL_SKY_VARS))

                self.observation_sources += len(temp_raw) * [train_file]

                if df_raw is None:
                    df_raw = temp_raw
                    df_all_sky = temp_all_sky
                else:
                    df_raw = pd.concat([df_raw, temp_raw], ignore_index=True)
                    df_all_sky = pd.concat([df_all_sky, temp_all_sky],
                                           ignore_index=True)

            logger.debug('\tShape temp_raw={}, temp_all_sky={}'
                         ''.format(temp_raw.shape, temp_all_sky.shape))
            time_step = calc_time_step(temp_raw.time_index)
            logger.debug('\tTime step is {} minutes'.format(time_step))

            # ------ Grab surface data
            year, _ = extract_file_meta(train_file)
            logger.debug('\tGrabbing surface data for {} and {}'
                         .format(year, self.train_sites))
            for gid in self.train_sites:
                code = surf_meta().loc[gid, 'surfrad_id']
                w_minutes = self._config.get('surfrad_window_minutes', 15)
                surfrad_file = self.fp_surfrad_data.format(year=year,
                                                           code=code)
                logger.debug('\t\tGrabbing surface data for {} from {}'
                            .format(code, surfrad_file))
                with Surfrad(surfrad_file) as surf:
                    temp_surf = surf.get_df(dt_out='{}min'.format(time_step),
                                            window_minutes=w_minutes)

                # add prefix to avoid confusion
                temp_surf = temp_surf.rename({'ghi': 'surfrad_ghi',
                                              'dni': 'surfrad_dni',
                                              'dhi': 'surfrad_dhi'}, axis=1)
                temp_surf['gid'] = gid
                temp_surf['time_index'] = temp_surf.index.values
                if df_surf is None:
                    df_surf = temp_surf
                else:
                    df_surf = pd.concat([df_surf, temp_surf],
                                        ignore_index=True)

                logger.debug('\tShape: temp_surf={}'.format(temp_surf.shape))

        logger.debug('Data load complete. Shape df_raw={}, df_all_sky={}, '
                     'df_surf={}'.format(df_raw.shape, df_all_sky.shape,
                                         df_surf.shape))

        self.observation_sources = np.array(self.observation_sources)
        assert df_raw.shape[0] == df_all_sky.shape[0]
        assert df_raw.shape[0] == df_surf.shape[0]
        assert len(self.observation_sources) == len(df_raw)
        assert all(df_all_sky.gid.values == df_surf.gid.values)
        assert all(df_all_sky.time_index.values == df_surf.time_index.values)
        time_index_full = df_all_sky.time_index
        df_surf.index = df_all_sky.index.values
        df_surf = df_surf.drop(['gid', 'time_index'], axis=1)
        df_all_sky = df_all_sky.join(df_surf)

        assert len(df_raw) == len(df_all_sky)
        if test_fraction:
            np.random.seed(self._config['phygnn_seed'])

            df_raw, df_all_sky = self._test_train_split(df_raw, df_all_sky,
                                                        time_index_full,
                                                        test_fraction)

        assert len(df_raw) == len(df_all_sky)

        logger.debug('Extracting 2D arrays to run rest2 for '
                     'clearsky PhyGNN inputs.')
        n = len(df_all_sky)
        time_index = pd.DatetimeIndex(df_all_sky.time_index.astype(str))
        aod = df_all_sky.aod.values.reshape((n, 1))
        alpha = df_all_sky.alpha.values.reshape((n, 1))
        surface_pressure = df_all_sky.surface_pressure.values.reshape((n, 1))
        surface_albedo = df_all_sky.surface_albedo.values.reshape((n, 1))
        ssa = df_all_sky.ssa.values.reshape((n, 1))
        asymmetry = df_all_sky.asymmetry.values.reshape((n, 1))
        solar_zenith_angle = (df_all_sky.solar_zenith_angle
                              .values.reshape((n, 1)))
        ozone = df_all_sky.ozone.values.reshape((n, 1))
        total_precipitable_water = (df_all_sky.total_precipitable_water
                                    .values.reshape((n, 1)))
        doy = time_index.dayofyear.values

        logger.debug('Running rest2 for clearsky PhyGNN inputs.')
        radius = ti_to_radius(time_index, n_cols=1)
        beta = calc_beta(aod, alpha)
        rest_data = rest2(surface_pressure, surface_albedo, ssa, asymmetry,
                          solar_zenith_angle,
                          radius, alpha, beta, ozone, total_precipitable_water)
        Tuuclr = rest2_tuuclr(surface_pressure, surface_albedo, ssa, radius,
                              alpha, ozone, total_precipitable_water,
                              parallel=False)

        df_all_sky['doy'] = doy
        df_all_sky['radius'] = radius
        df_all_sky['Tuuclr'] = Tuuclr
        df_all_sky['clearsky_ghi'] = rest_data.ghi
        df_all_sky['clearsky_dni'] = rest_data.dni
        df_all_sky['Ruuclr'] = rest_data.Ruuclr
        df_all_sky['Tddclr'] = rest_data.Tddclr
        df_all_sky['Tduclr'] = rest_data.Tduclr
        logger.debug('Completed rest2 run for clearsky PhyGNN inputs.')

        self.df_raw = df_raw

        # Temporarily extract time_index or interpolate will break
        time_index = df_all_sky.time_index
        assert time_index.isnull().sum() == 0
        df_all_sky = df_all_sky.drop('time_index', axis=1)
        self.df_all_sky = df_all_sky.interpolate('nearest').bfill().ffill()
        self.df_all_sky['time_index'] = time_index
        assert len(df_raw) == len(df_all_sky)

    def _prep_data(self, kwargs=TRAINING_PREP_KWARGS):
        """
        Clean and prepare training data

        Parameters
        ----------
        kwargs: dict
            Keyword arguments for clean_cloud_df()
        """
        logger.debug('Training data clean kwargs: {}'.format(kwargs))
        logger.debug('Shape before cleaning: df_raw={}'
                     ''.format(self.df_raw.shape))
        self.df_train = clean_cloud_df(self.df_raw, **kwargs)
        logger.debug('Shape after cleaning: df_train={}'
                     ''.format(self.df_train.shape))

        logger.debug('Cleaning df_all_sky training data (for pfun).')
        logger.debug('Shape before cleaning: df_all_sky={}'
                     ''.format(self.df_all_sky.shape))

        self.df_all_sky = clean_cloud_df(self.df_all_sky, **kwargs)
        logger.debug('Shape after cleaning: df_all_sky={}'
                     ''.format(self.df_all_sky.shape))

        # Inspecting features would go here

        # Final cleaning
        drop_list = ['gid', 'time_index', 'cloud_type']
        if self._config.get('one_hot_categories', None) is None:
            drop_list.append('flag')

        for name in drop_list:
            if name in self.df_train:
                self.df_train = self.df_train.drop(name, axis=1)

        logger.debug('**Shape: df_train={}'.format(self.df_train.shape))
        features = self.df_train.columns.values.tolist()

        not_features = drop_list + list(self._config['y_labels'])
        features = [f for f in features if f not in not_features]

        self.y = self.df_train[self._config['y_labels']]
        self.x = self.df_train[features]
        self.p = self.df_all_sky

        logger.debug('Shapes: x={}, y={}, p={}'.format(self.x.shape,
                                                       self.y.shape,
                                                       self.p.shape))
        logger.debug('Training features: {}'.format(features))
        assert self.y.shape[0] == self.x.shape[0] == self.p.shape[0]

    def _test_train_split(self, df_raw_orig, df_all_sky_orig, time_index_full,
                          test_fraction):
        """
        Split data into test and train sets. Return train data.

        Parameters
        ----------
        df_raw_orig: pandas.DataFrame
            Satellite data for model training
        df_all_sky_orig: pandas.DataFrame
            All_sky inputs
        time_index_full : pd.DatetimeIndex
            Time index corresponding to the df_all_sky_orig input.
        test_fraction: None | float
            Fraction of full data set to reserve for testing. Should be between
            0 to 1. The test set is randomly selected and dropped from the
            training set. If None, do not reserve a test set.

        Returns
        -------
        df_raw: pandas.DataFrame
            Training set of satellite data for model training
        df_all_sky: pandas.DataFrame
            Training set of all_sky inputs
        """
        logger.debug('Creating test set; {}% of full data set'
                     ''.format(test_fraction*100))
        assert 0 < test_fraction < 1
        assert len(time_index_full) == len(df_all_sky_orig)

        ti1 = df_raw_orig['time_index'].values
        ti2 = df_all_sky_orig['time_index'].values
        msg = ('Time indices dont match, something went wrong: \n{} \n{}'
               .format(ti1, ti2))
        assert (ti1 == ti2).all(), msg

        df_raw = df_raw_orig.sample(frac=(1-test_fraction)).sort_index()
        self.train_set_mask = df_raw_orig.index.isin(df_raw.index)
        df_all_sky = df_all_sky_orig[self.train_set_mask]
        self.test_set_mask = ~self.train_set_mask

        logger.debug('Train set shape: df_raw={}, df_all_sky={}'
                     ''.format(df_raw.shape, df_all_sky.shape))
        logger.debug('Test set shape: df_raw={}, df_all_sky={}'
                     ''.format(df_raw_orig[self.test_set_mask].shape,
                               df_all_sky_orig[self.test_set_mask].shape))
        return df_raw, df_all_sky

    @property
    def all_data(self):
        """Get a joined dataframe of training data (x), physical feature data
        (p), and output label data (y).

        Returns
        -------
        pd.DataFrame
        """
        cols_p = [c for c in self.p.columns if c not in self.x]
        out = self.x.join(self.p[cols_p])
        cols_y = [c for c in self.y.columns if c not in out]
        out = out.join(self.y[cols_y])
        return out

    def save_all_data(self, fp):
        """Save all x/y/p data to disk

        Parameters
        ----------
        fp : str
            .csv filepath to save data to.
        """
        if fp is not None:
            logger.info('Saving training data to: {}'.format(fp))
            self.all_data.to_csv(fp)


class ValidationData:
    """ Load and prep validation data """

    def __init__(self, val_files, features=CONFIG['features'],
                 y_labels=CONFIG['y_labels'], all_sky_vars=ALL_SKY_VARS,
                 one_hot_cats=None, predict_clearsky=True, test_set_mask=None):
        """
        Parameters
        ----------
        val_files: str | list of str
            List of files to use for validation
        features: list of str
            Names of model input fields
        y_labels: list of str
            Names of model output fields
        all_sky_vars: list of str
            Names of fields used for the allsky algorithm
        one_hot_cats: dict | None
            Categories for one hot encoding. Keys are column names, values
            are lists of category values. See phygnn.utlities.pre_processing.
        predict_clearsky: bool
            Let phygnn predict properties for clear and cloudy time steps if
            true, else, only predict properties for cloudy time steps.
        test_set_mask: None | numpy.ndarray of bool
            Set of full data set in val_files to use. If None, use full
            dataset.
        """
        self.means = 0
        self.stdevs = 1
        self.fp_surfrad_meta = FP_SURFRAD_META
        self.features = features
        self.y_labels = y_labels
        self.all_sky_vars = all_sky_vars
        self.one_hot_cats = one_hot_cats

        # dict of year, time_step, and time_step info for val_files
        self.files_meta = []

        if isinstance(val_files, str):
            val_files = [val_files]
        self.val_files = val_files

        self._load_data(test_set_mask)
        self._prep_data(predict_clearsky)

    def _load_data(self, test_set_mask):
        """
        Load validation data

        Parameters
        ----------
        test_set_mask: None | numpy.ndarray of bool
            Set of full data set in val_files to use. If None, use full
            dataset.
        """
        logger.debug('Loading validation data')

        df_raw = None
        df_all_sky = None
        var_names = copy.deepcopy(self.features)
        var_names += self.y_labels
        logger.debug('Loading vars {}'.format(var_names))

        gids = [k for k, v in surf_meta().to_dict()['surfrad_id'].items()]

        for val_file in self.val_files:
            logger.debug('Loading validation data from {} for gids {}'
                         ''.format(val_file, gids))
            with NSRDBFeatures(val_file) as res:
                temp_raw = res.extract_features(gids, var_names)
                temp_all_sky = res.extract_features(
                    gids, self.all_sky_vars)
                if df_raw is None:
                    df_raw = temp_raw
                    df_all_sky = temp_all_sky
                else:
                    df_raw = pd.concat([df_raw, temp_raw], ignore_index=True)
                    df_all_sky = pd.concat([df_all_sky, temp_all_sky],
                                           ignore_index=True)

            year, area = extract_file_meta(val_file)
            time_step = calc_time_step(temp_raw.time_index)
            self.files_meta.append({'year': year,
                                    'area': area,
                                    'time_step': time_step})
            logger.debug('\tShape temp_raw={}, temp_all_sky={}, & tstep={} '
                         'minutes'.format(temp_raw.shape, temp_all_sky.shape,
                                          time_step))
        logger.debug('Shape df_raw={}, df_all_sky={}'
                     ''.format(df_raw.shape, df_all_sky.shape))

        assert df_raw.shape[0] == df_all_sky.shape[0]
        df_raw.reset_index(drop=True, inplace=True)
        df_all_sky.reset_index(drop=True, inplace=True)

        logger.debug('Shape after reset_index: df_raw={}, df_all_sky={}'
                     ''.format(df_raw.shape, df_all_sky.shape))

        if test_set_mask is not None:
            assert (len(df_raw) == len(df_all_sky) == len(test_set_mask)), \
                ('test_set_mask is the wrong length: {}. Ensure same sites '
                 'and files are used for training and validation.'
                 '').format(len(test_set_mask))
            df_raw = df_raw[test_set_mask]
            df_all_sky = df_all_sky[test_set_mask]
            logger.debug('Test set shape: df_raw={}, df_all_sky={}'
                         ''.format(df_raw.shape, df_all_sky.shape))

        # we filter daylight and clear later
        self.df_feature_val = clean_cloud_df(df_raw, filter_daylight=False,
                                             filter_clear=False,
                                             add_cloud_flag=True,
                                             sza_lim=89)
        self.df_all_sky_val = clean_cloud_df(df_all_sky, filter_daylight=False,
                                             filter_clear=False,
                                             add_cloud_flag=True,
                                             sza_lim=89)

    def _prep_data(self, predict_clearsky):
        """
        Prepare validation data

        Parameters
        ----------
        predict_clearsky: bool
            Let phygnn predict properties for clear and cloudy time steps if
            true, else, only predict properties for cloudy time steps.
        """
        logger.debug('Prepping validation data')

        day_mask = (self.df_feature_val['solar_zenith_angle'] < 89)
        cloud_mask = (day_mask & self.df_feature_val['cloud_type']
                      .isin(ICE_TYPES + WATER_TYPES))

        if predict_clearsky:
            self.mask = day_mask  # let phygnn predict clearsky
        else:
            self.mask = cloud_mask  # let phygnn predict only clouds
        logger.debug('Mask: shape={}, sum={}'.format(self.mask.shape,
                                                     self.mask.sum()))

        drop_list = ['gid', 'time_index', 'cloud_type']
        not_features = drop_list + list(self.y_labels)
        if self.one_hot_cats is None:
            not_features.append('flag')

        features = [c for c in self.df_feature_val.columns if
                    c not in not_features]
        self.df_x_val = self.df_feature_val.loc[self.mask, features]
        logger.debug('Validation features: {}'.format(features))
