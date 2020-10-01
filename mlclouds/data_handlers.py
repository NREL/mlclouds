import logging
import copy
import pandas as pd

from nsrdb.all_sky.rest2 import rest2, rest2_tuuclr
from nsrdb.all_sky.utilities import ti_to_radius, calc_beta
from nsrdb.all_sky import ICE_TYPES, WATER_TYPES
from nsrdb.file_handlers.surfrad import Surfrad

from phygnn.utilities import PreProcess

from mlclouds.data_cleaners import clean_cloud_df
from mlclouds.nsrdb import NSRDBFeatures
from mlclouds.utilities import ALL_SKY_VARS, CONFIG
from mlclouds.utilities import FP_SURFRAD_DATA, FP_SURFRAD_META
from mlclouds.utilities import TRAINING_PREP_KWARGS
from mlclouds.utilities import surf_meta, calc_time_step, extract_file_meta

logger = logging.getLogger(__name__)


class TrainData:
    """ Load and prep training data """
    def __init__(self, train_sites, train_files, config=CONFIG):
        """
        Parameters
        ----------
        train_sites: list of int
            Surfrad gids to use for training
        train_files: list | str
            File or list of files to use for training. Filenames must include
            the four-digit year.
        config: dict
            Dict of configuration options. See CONFIG for example.
        """
        self.fp_surfrad_data = FP_SURFRAD_DATA
        self.fp_surfrad_meta = FP_SURFRAD_META
        self.train_sites = train_sites
        self._config = config

        self.means = None
        self.stdevs = None

        if not isinstance(train_files, list):
            train_files = [train_files]
        self.train_files = train_files

        logger.info('Loading training data')
        self.load_data()
        logger.info('Prepping training data')
        self.prep_data()

    def load_data(self):
        """ Load training data """
        var_names = copy.deepcopy(self._config['features'])
        var_names += self._config['y_labels']
        logger.debug('Loading vars {}'.format(var_names))

        # Grab satellite and ground data for training files
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

                if df_raw is None:
                    df_raw = temp_raw
                    df_all_sky = temp_all_sky
                else:
                    df_raw = df_raw.append(temp_raw, ignore_index=True)
                    df_all_sky = df_all_sky.append(temp_all_sky,
                                                   ignore_index=True)

            logger.debug('\tShape temp_raw={}, temp_all_sky={}'
                         ''.format(temp_raw.shape, temp_all_sky.shape))
            time_step = calc_time_step(temp_raw.time_index)
            logger.debug('\tTime step is {} minutes'.format(time_step))
            assert df_raw.shape[0] == df_all_sky.shape[0]

            # ------ Grab surface data
            year, _ = extract_file_meta(train_file)
            logger.debug('\tGrabbing surface data for {} and {}'
                         .format(year, self.train_sites))
            for gid in self.train_sites:
                code = surf_meta().loc[gid, 'surfrad_id']
                w_minutes = self._config.get('surfrad_window_minutes', 15)
                surfrad_file = self.fp_surfrad_data.format(year=year,
                                                           code=code)
                with Surfrad(surfrad_file) as surf:
                    temp_surf = surf.get_df(dt_out='{}min'.format(time_step),
                                            window_minutes=w_minutes)
                temp_surf['gid'] = gid
                temp_surf['time_index'] = temp_surf.index.values
                if df_surf is None:
                    df_surf = temp_surf
                else:
                    df_surf = df_surf.append(temp_surf, ignore_index=True)

                logger.debug('\tShape: temp_surf={}'.format(temp_surf.shape))
        logger.debug('Shape df_raw={}, df_all_sky={}, df_surf={}'
                     ''.format(df_raw.shape, df_all_sky.shape, df_surf.shape))

        assert all(df_all_sky.gid.values == df_surf.gid.values)
        assert all(df_all_sky.time_index.values == df_surf.time_index.values)

        df_surf = df_surf.reset_index(drop=True)
        df_surf = df_surf.drop(['gid', 'time_index'], axis=1)
        df_all_sky = df_all_sky.join(df_surf)

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

    def prep_data(self, kwargs=TRAINING_PREP_KWARGS):
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

        # Final cleaning and one-hot encoding
        drop_list = ('gid', 'time_index', 'cloud_type')
        for name in drop_list:
            if name in self.df_train:
                self.df_train = self.df_train.drop(name, axis=1)
        logger.debug('Adding one-hot vectors to training data.')
        logger.debug('*Shape: df_train={}'.format(self.df_train.shape))

        self.df_train = PreProcess.one_hot(
            self.df_train, categories=self._config['one_hot_categories'])
        one_hot_labels = [a for b in
                          self._config['one_hot_categories'].values()
                          for a in b]
        norm_cols = [c for c in self.df_train.columns
                     if c not in one_hot_labels
                     and c not in self._config['y_labels']]
        self.df_train[norm_cols], means, stdevs = PreProcess.normalize(
            self.df_train[norm_cols])
        self.means = means
        self.stdevs = stdevs
        temp_means = {c: means[i] for i, c in enumerate(norm_cols)}
        temp_stdevs = {c: stdevs[i] for i, c in enumerate(norm_cols)}
        logger.debug('Norm means: {}'.format(temp_means))
        logger.debug('Norm stdevs: {}'.format(temp_stdevs))

        logger.debug('**Shape: df_train={}'.format(self.df_train.shape))
        features = self.df_train.columns.values.tolist()

        not_features = drop_list + tuple(self._config['y_labels'])
        features = [f for f in features if f not in not_features]

        self.y = self.df_train[self._config['y_labels']]
        self.x = self.df_train[features]
        self.p = self.df_all_sky

        logger.debug('Shapes: x={}, y={}, p={}'.format(self.x.shape,
                                                       self.y.shape,
                                                       self.p.shape))
        logger.debug('Training features: {}'.format(features))
        assert self.y.shape[0] == self.x.shape[0] == self.p.shape[0]


class ValidationData:
    """ Load and prep validation data """

    def __init__(self, val_files, features, y_labels,
                 all_sky_vars=ALL_SKY_VARS, one_hot_cats=None,
                 predict_clearsky=True):
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

        self.load_data()
        self.prep_data(predict_clearsky)

    def load_data(self):
        """ Load validation data """
        logger.debug('Loading validation data')

        df_raw = None
        df_all_sky = None
        var_names = copy.deepcopy(self.features)
        var_names += self.y_labels
        logger.debug('Loading vars {}'.format(var_names))

        all_gids = surf_meta().index.unique().values.tolist()
        for val_file in self.val_files:
            logger.debug('Loading validation data from {} for gids {}'
                         ''.format(val_file, all_gids))
            with NSRDBFeatures(val_file) as res:
                temp_raw = res.extract_features(all_gids, var_names)
                temp_all_sky = res.extract_features(
                    all_gids, self.all_sky_vars)
                if df_raw is None:
                    df_raw = temp_raw
                    df_all_sky = temp_all_sky
                else:
                    df_raw = df_raw.append(temp_raw)
                    df_all_sky = df_all_sky.append(temp_all_sky)

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

        self.df_feature_val = clean_cloud_df(df_raw, filter_daylight=False,
                                             filter_clear=False,
                                             add_feature_flag=True, sza_lim=89)
        self.df_all_sky_val = clean_cloud_df(df_all_sky, filter_daylight=False,
                                             filter_clear=False,
                                             add_feature_flag=True, sza_lim=89)

    def prep_data(self, predict_clearsky):
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

        drop_list = ('gid', 'time_index', 'cloud_type')
        not_features = drop_list + tuple(self.y_labels)

        features = [c for c in self.df_feature_val.columns if
                    c not in not_features]
        self.df_x_val = PreProcess.one_hot(
            self.df_feature_val.loc[self.mask, features],
            categories=self.one_hot_cats)
        logger.debug('Validation features: {}'.format(features))

    def norm_data(self, means, stdevs):
        """ Normalize the feature inputs for the validation dataset. """
        self.means = means
        self.stdevs = stdevs
        one_hot_labels = [a for b in self.one_hot_cats.values()
                          for a in b]
        norm_cols = [c for c in self.df_x_val.columns
                     if c not in one_hot_labels
                     and c not in self.y_labels]
        self.df_x_val[norm_cols], m, s = PreProcess.normalize(
            self.df_x_val[norm_cols], mean=means, stdev=stdevs)
        temp_means = {c: m[i] for i, c in enumerate(norm_cols)}
        temp_stdevs = {c: s[i] for i, c in enumerate(norm_cols)}
        logger.debug('Norm means: {}'.format(temp_means))
        logger.debug('Norm stdevs: {}'.format(temp_stdevs))

    def un_norm_data(self, means=None, stdevs=None):
        """ Un-normalize the feature inputs for the validation dataset. """
        if means is None:
            means = self.means
        if stdevs is None:
            stdevs = self.stdevs
        one_hot_labels = [a for b in self.one_hot_cats.values()
                          for a in b]
        norm_cols = [c for c in self.df_x_val.columns
                     if c not in one_hot_labels
                     and c not in self.y_labels]
        logger.debug('Un-normalizing validation feature inputs.')
        self.df_x_val[norm_cols] = PreProcess.unnormalize(
            self.df_x_val[norm_cols], means, stdevs)
