"""
Automatic cross validation of PhyGNN models predicting opd and reff

Mike Bannister 7/2020
Based on code by Grant Buster
"""
import pandas as pd
import numpy as np
import os
import time
import logging
import plotly.express as px

import tensorflow as tf

from rex.resource import MultiFileResource

from nsrdb.file_handlers.surfrad import Surfrad
from nsrdb.all_sky.all_sky import all_sky
from nsrdb.all_sky import ICE_TYPES, WATER_TYPES
from nsrdb.all_sky.rest2 import rest2, rest2_tuuclr
from nsrdb.all_sky.utilities import ti_to_radius, calc_beta
from nsrdb.utilities.statistics import mae_perc, mbe_perc, rmse_perc

from mlclouds.nsrdb import NSRDBFeatures
from mlclouds.tdisc import tdisc
from mlclouds.tfarms import tfarms

from phygnn import PhysicsGuidedNeuralNetwork as PGNN
from phygnn.pre_processing import PreProcess

logger = logging.getLogger(__name__)


def p_fun_all_sky(y_predicted, y_true, p, labels=None):
    """ Physics loss function """
    n = len(y_true)
    tau = tf.expand_dims(y_predicted[:, 0], axis=1)
    cld_reff = tf.expand_dims(y_predicted[:, 1], axis=1)

    tau = tf.where(tau < 0, 0.0001, tau)
    tau = tf.where(tau > 160, 160, tau)
    cld_reff = tf.where(cld_reff < 0, 0.0001, cld_reff)
    cld_reff = tf.where(cld_reff > 160, 160, cld_reff)

    cloud_type = p[:, labels.index('cloud_type')
                   ].astype(np.float32).reshape((n, 1))
    solar_zenith_angle = p[:, labels.index('solar_zenith_angle')
                           ].astype(np.float32).reshape((n, 1))
    doy = p[:, labels.index('doy')].astype(np.float32).reshape((n, 1))
    radius = p[:, labels.index('radius')].astype(np.float32).reshape((n, 1))
    Tuuclr = p[:, labels.index('Tuuclr')].astype(np.float32).reshape((n, 1))
    Ruuclr = p[:, labels.index('Ruuclr')].astype(np.float32).reshape((n, 1))
    Tddclr = p[:, labels.index('Tddclr')].astype(np.float32).reshape((n, 1))
    Tduclr = p[:, labels.index('Tduclr')].astype(np.float32).reshape((n, 1))
    albedo = p[:, labels.index('surface_albedo')
               ].astype(np.float32).reshape((n, 1))
    pressure = p[:, labels.index('surface_pressure')
                 ].astype(np.float32).reshape((n, 1))

    cs_dni = p[:, labels.index('clearsky_dni')].astype(np.float32).reshape((n,
                                                                            1))
    cs_ghi = p[:, labels.index('clearsky_ghi')].astype(np.float32).reshape((n,
                                                                            1))
    dni_ground = p[:, labels.index('dni')].astype(np.float32).reshape((n, 1))
    ghi_ground = p[:, labels.index('ghi')].astype(np.float32).reshape((n, 1))
    cs_dni = tf.convert_to_tensor(cs_dni, dtype=tf.float32)
    cs_ghi = tf.convert_to_tensor(cs_ghi, dtype=tf.float32)
    dni_ground = tf.convert_to_tensor(dni_ground, dtype=tf.float32)
    ghi_ground = tf.convert_to_tensor(ghi_ground, dtype=tf.float32)

    ghi_predicted = tfarms(tau, cloud_type, cld_reff, solar_zenith_angle,
                           radius, Tuuclr, Ruuclr, Tddclr, Tduclr, albedo)
    dni_predicted = tdisc(ghi_predicted, solar_zenith_angle, doy,
                          pressure=pressure)

    dni_predicted = tf.where(tau == 0.0001, cs_dni, dni_predicted)
    ghi_predicted = tf.where(tau == 0.0001, cs_ghi, ghi_predicted)

    err_ghi = ghi_predicted - ghi_ground
    err_ghi = tf.boolean_mask(err_ghi, ~tf.math.is_nan(err_ghi))
    err_ghi = tf.boolean_mask(err_ghi, tf.math.is_finite(err_ghi))

    err_dni = dni_predicted - dni_ground
    err_dni = tf.boolean_mask(err_dni, ~tf.math.is_nan(err_dni))
    err_dni = tf.boolean_mask(err_dni, tf.math.is_finite(err_dni))

    mae_ghi = tf.reduce_mean(tf.abs(err_ghi)) / tf.reduce_mean(ghi_ground)
    mae_dni = tf.reduce_mean(tf.abs(err_dni)) / tf.reduce_mean(dni_ground)
    mbe_ghi = tf.abs(tf.reduce_mean(err_ghi)) / tf.reduce_mean(ghi_ground)
    mbe_dni = tf.abs(tf.reduce_mean(err_dni)) / tf.reduce_mean(dni_ground)

    p_loss = mae_ghi + mae_dni + mbe_ghi + mbe_dni
    return p_loss


CONFIG = {
    'fp_data': '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/{year}_adj/'
               'mlclouds_surfrad_{year}_adj.h5',
    'fp_surf': '/lustre/eaglefs/projects/mlclouds/ground_measurement/surfrad/'
               '{code}_{year}.h5',
    'model_dir': './models',
    'fp': '/lustre/eaglefs/projects/mlclouds/ground_measurement/'
          'surfrad_meta.csv',

    'features': ['solar_zenith_angle', 'cloud_type',
                 'refl_0_65um_nom', 'refl_3_75um_nom',
                 'temp_3_75um_nom', 'cloud_transmission_0_65um_nom',
                 'cloud_fraction', 'air_temperature',
                 'dew_point', 'relative_humidity',
                 'total_precipitable_water', 'surface_albedo',
                 'cld_opd_dcomp', 'cld_reff_dcomp',
                 ],

    'all_sky_vars': ['alpha', 'aod', 'asymmetry', 'cloud_type',
                     'cld_opd_dcomp',
                     'cld_reff_dcomp', 'ozone', 'solar_zenith_angle', 'ssa',
                     'surface_albedo', 'surface_pressure',
                     'total_precipitable_water',
                     ],

    # TODO - the 'y's being used are hardwired in the validation code
    'y_labels': ['cld_opd_dcomp', 'cld_reff_dcomp'],

    # TODO - There is some overlap and confusion between the next two
    # parameters Fields to drop from training data before preprocessing
    'drop_list': ['gid', 'time_index', 'cloud_type'],

    # Fields to exclude from training data (x)
    'ignore': ('gid', 'time_index', 'cloud_type', 'cld_opd_dcomp',
               'cld_reff_dcomp',
               ),

    # Neural network geometry
    'hidden_layers': [{'units': 64, 'activation': 'relu', 'name': 'relu1',
                       'dropout': 0.01},
                      {'units': 64, 'activation': 'relu', 'name': 'relu2',
                       'dropout': 0.01},
                      {'units': 64, 'activation': 'relu', 'name': 'relu3',
                       'dropout': 0.01},
                      ],

    # phygnn params
    'phygnn_seed': 0,
    'p_fun': p_fun_all_sky,
    'metric': 'relative_mae',
    'learning_rate': 0.01,
    'n_batch': 4,

    # Two training stages are used, with independent number of epochs and
    # loss weights. The loss weight tuple is (pure nn loss, physics loss). By
    # default the first stage of training ignores the physics loss and the
    # second stage weights the two losses evenly.
    'epochs_a': 100,
    'epochs_b': 100,
    'loss_weights_a': (1, 0),
    'loss_weights_b': (1, 1),

    # Validation vars
    'fp_baseline': '/lustre/eaglefs/projects/mlclouds/'
                   'data_surfrad_9/{year}/final/srf{yy}_*_{year}.h5',
    'fp_baseline_adj': '/lustre/eaglefs/projects/mlclouds/'
                       'data_surfrad_9/{year}_adj/final/srf{yy}a_*_{year}.h5',
}


class XVal:
    """
    Train a phygnn using selected sites then validate against an excluded
    site.
    """

    def __init__(self, train_sites=[0, 1, 2, 3, 5, 6], val_site=4,
                 years=(2018,), config=CONFIG, val_data=None,
                 save_model=False,
                 load_model=False):
        """
        Parameters
        ----------
        train_sites: list
            Sites to use for training
        val_site: int
            Site to use for validation
        years: tuple
            Years of data to use
        config: dict
            Dict of configuration options. See CONFIG for example.
        val_data: None | ValidationData instance
            Use preloaded validation data or load if None
        save_model: bool
            Save trained model to disc as pickle file
        load_model: bool
            Skip training and load model from pickle file
        """
        logger.info('XV: Training on {}, validating against {}, for {}'
                    ''.format(train_sites, val_site, years))
        assert val_site not in train_sites

        self.train_sites = train_sites
        self.val_site = val_site
        self.config = config

        if isinstance(years, int):
            years = (years,)
        self.years = years

        if not os.path.exists(self.config['model_dir']):
            os.makedirs(self.config['model_dir'])

        surf_meta = pd.read_csv(self.config['fp'], index_col=0)
        surf_meta.index.name = 'gid'
        self.surf_meta = surf_meta[['surfrad_id']]

        # Train or load phygnn model
        if not load_model:
            train_data = TrainData(self.years, self.train_sites, self.config)
            self.x = train_data.x
            self.y = train_data.y
            self.p = train_data.p
            self.p_kwargs = train_data.p_kwargs
            self.train()

            if save_model:
                self.model.save(self.model_file)
        else:
            logger.debug('Loading model from {}'.format(self.model_file))
            self.model = PGNN.load(self.model_file)

        # Validate model
        if val_data is None:
            val_data = ValidationData(years, config)

        self.df_x_val = val_data.df_x_val
        self.df_all_sky_val = val_data.df_all_sky_val
        # TODO - df_feature_val appears to only be used for mask generation,
        # is that right?
        self.df_feature_val = val_data.df_feature_val
        self.mask = val_data.mask
        self.validate()
        self.calc_stats()

    def train(self):
        """ Train PhyGNN model """
        # TODO - update to control xfer learning via config dict
        logger.debug('Building PhyGNN model')
        PGNN.seed(self.config['phygnn_seed'])
        model = PGNN(p_fun=self.config['p_fun'],
                     hidden_layers=self.config['hidden_layers'],
                     loss_weights=self.config['loss_weights_a'],
                     metric=self.config['metric'],
                     input_dims=self.x.shape[1],
                     output_dims=self.y.shape[1],
                     learning_rate=self.config['learning_rate'])

        logger.info('Training part A - pure data. Loss is {}'
                    ''.format(self.config['loss_weights_a']))
        out = model.fit(self.x, self.y, self.p, n_batch=self.config['n_batch'],
                        n_epoch=self.config['epochs_a'],
                        p_kwargs=self.p_kwargs)

        logger.info('Training part B - data and phygnn. Loss is {}'
                    ''.format(self.config['loss_weights_b']))
        model.set_loss_weights(self.config['loss_weights_b'])
        out = model.fit(self.x, self.y, self.p, n_batch=self.config['n_batch'],
                        n_epoch=self.config['epochs_b'],
                        p_kwargs=self.p_kwargs)

        model.save(self.model_file)

        self.model = model
        self.out = out

    def validate(self, update_clear=False, update_cloudy=False):
        """ Validate PhyGNN model predictions """
        logger.info('Validating model')
        t0 = time.time()

        # Predict opd and reff using model
        predicted_raw = self.model.predict(self.df_x_val.values)
        logger.debug('Prediction took {:.2f} seconds'.format(time.time() - t0))
        opd_raw = predicted_raw[:, 0]
        reff_raw = predicted_raw[:, 1]

        opd = np.minimum(opd_raw, 160)
        reff = np.minimum(reff_raw, 160)
        opd = np.maximum(opd, 0.0)
        reff = np.maximum(reff, 0.0)

        # opd_orig = self.df_all_sky_val['cld_opd_dcomp'].copy()
        # reff_orig = self.df_all_sky_val['cld_reff_dcomp'].copy()

        self.df_feature_val['cld_opd_dcomp'] = 0
        self.df_feature_val['cld_reff_dcomp'] = 0
        self.df_feature_val.loc[self.mask, 'cld_opd_dcomp'] = opd
        self.df_feature_val.loc[self.mask, 'cld_reff_dcomp'] = reff

        self.df_all_sky_val['cld_opd_dcomp'] = 0
        self.df_all_sky_val['cld_reff_dcomp'] = 0
        self.df_all_sky_val.loc[self.mask, 'cld_opd_dcomp'] = opd
        self.df_all_sky_val.loc[self.mask, 'cld_reff_dcomp'] = reff

        if update_clear:
            mask = ((self.df_feature_val['solar_zenith_angle'] < 89)
                    & (self.df_feature_val['cld_opd_dcomp'] <= 0.0)
                    & self.df_feature_val['cloud_type'].isin(ICE_TYPES
                                                             + WATER_TYPES))
            self.df_feature_val.loc[mask, 'cloud_type'] = 0
            self.df_all_sky_val.loc[mask, 'cloud_type'] = 0
            logger.debug('The PGNN predicted {} additional clear timesteps'
                         '({:.2f}%)'.format(mask.sum(),
                                            100 * mask.sum() / len(mask)))
        if update_cloudy:
            mask = ((self.df_feature_val['solar_zenith_angle'] < 89)
                    & (self.df_feature_val['cld_opd_dcomp'] > 0.0)
                    & (self.df_feature_val['cloud_type'] <= 1))
            self.df_feature_val.loc[mask, 'cloud_type'] = 8
            self.df_all_sky_val.loc[mask, 'cloud_type'] = 8
            print('The PGNN predicted {} additional cloudy timesteps ({:.2f}%)'
                  ''.format(mask.sum(), 100 * mask.sum() / len(mask)))

        return

    def calc_stats(self):
        """ Calculate accuracy of PhyGNN model predictions """
        logger.info('Calculating statistics')
        all_sky_outs = {}
        stats = pd.DataFrame(columns=['Model', 'Site', 'Variable',
                                      'Condition'])
        i = 0

        all_sky_gids = [k for k, v in
                        self.surf_meta.to_dict()['surfrad_id'].items()
                        if v not in ['srrl', 'sgp']]

        for gid in all_sky_gids:
            code = self.surf_meta.loc[gid, 'surfrad_id']
            logger.debug('Computing stats for gid: {} {}'
                         .format(gid, code))

            logger.info('Loading data for stats')
            df_baseline, df_baseline_adj, df_surf = self._get_stats_data(
                self.years, gid, code)
            logger.info('Loading data for stats complete')

            args = self._get_all_sky_args(gid, self.config['all_sky_vars'],
                                          self.df_all_sky_val)

            out = all_sky(**args)
            all_sky_out = pd.DataFrame({k: v.flatten() for k,
                                       v in out.items()},
                                       index=args['time_index'])
            all_sky_outs[code] = all_sky_out

            gid_mask = (self.df_all_sky_val.gid == gid)
            val_daylight_mask = (self.df_all_sky_val.loc[gid_mask,
                                 'solar_zenith_angle'] < 89).values
            val_cloudy_mask = (val_daylight_mask
                               & (self.df_all_sky_val
                                  .loc[gid_mask, 'cloud_type']
                                  .isin(ICE_TYPES + WATER_TYPES)).values)
            val_clear_mask = (val_daylight_mask
                              & (self.df_all_sky_val
                                 .loc[gid_mask, 'cloud_type'] <= 1).values)
            val_bad_cloud_mask = (val_cloudy_mask
                                  & (self.df_all_sky_val
                                     .loc[gid_mask, 'flag'] == 'bad_cloud')
                                  .values)

            m_iter = zip([val_daylight_mask, val_cloudy_mask,
                          val_clear_mask, val_bad_cloud_mask],
                         ['All-Sky', 'Cloudy', 'Clear',
                          'Missing Cloud Data'])

            for mask, condition in m_iter:
                for var in ['dni', 'ghi']:

                    baseline = df_baseline.loc[mask, var]
                    adjusted = df_baseline_adj.loc[mask, var]
                    surf = df_surf.loc[mask, var]
                    mlclouds = all_sky_out.loc[mask, var]

                    mae_baseline = mae_perc(baseline, surf)
                    mbe_baseline = mbe_perc(baseline, surf)
                    rmse_baseline = rmse_perc(baseline, surf)

                    mae_adj = mae_perc(adjusted, surf)
                    mbe_adj = mbe_perc(adjusted, surf)
                    rmse_adj = rmse_perc(adjusted, surf)

                    mae_ml = mae_perc(mlclouds, surf)
                    mbe_ml = mbe_perc(mlclouds, surf)
                    rmse_ml = rmse_perc(mlclouds, surf)

                    stats.at[i, 'Model'] = 'Baseline'
                    stats.at[i, 'Site'] = code.upper()
                    stats.at[i, 'Variable'] = var.upper()
                    stats.at[i, 'Condition'] = condition
                    stats.at[i, 'MAE (%)'] = mae_baseline
                    stats.at[i, 'MBE (%)'] = mbe_baseline
                    stats.at[i, 'RMSE (%)'] = rmse_baseline
                    i += 1

                    stats.at[i, 'Model'] = 'Adjusted'
                    stats.at[i, 'Site'] = code.upper()
                    stats.at[i, 'Variable'] = var.upper()
                    stats.at[i, 'Condition'] = condition
                    stats.at[i, 'MAE (%)'] = mae_adj
                    stats.at[i, 'MBE (%)'] = mbe_adj
                    stats.at[i, 'RMSE (%)'] = rmse_adj
                    i += 1

                    stats.at[i, 'Model'] = 'PhyGNN'
                    stats.at[i, 'Site'] = code.upper()
                    stats.at[i, 'Variable'] = var.upper()
                    stats.at[i, 'Condition'] = condition
                    stats.at[i, 'MAE (%)'] = mae_ml
                    stats.at[i, 'MBE (%)'] = mbe_ml
                    stats.at[i, 'RMSE (%)'] = rmse_ml
                    i += 1
        self.stats = stats

    def save_stats(self):
        """ Save statistics to file """
        train_name = ''.join([str(x) for x in self.train_sites])
        f_name = 'val_stats_{}_{}.csv'.format(train_name, self.val_site)
        f_name = os.path.join(self.config['model_dir'], f_name)
        self.stats.to_csv(f_name, index=False)

    def plot(self):
        """ Show statistics bar charts """
        gid = self.val_site
        code = self.surf_meta.loc[gid, 'surfrad_id']
        logger.debug('Plotting for {} {}'.format(self.val_site, code))

        for ylabel in ['MBE (%)', 'MAE (%)', 'RMSE (%)']:
            fig = px.bar(self.stats[(self.stats.Site == code.upper())],
                         x="Condition", y=ylabel, color='Model',
                         facet_col="Variable", barmode='group', height=400)
            fig.show()

    @property
    def model_file(self):
        """ Model file name including training and validation sites """
        train_name = ''.join([str(x) for x in self.train_sites])
        pkl_name = 'pgnn_{}_{}.pkl'.format(train_name, self.val_site)
        return os.path.join(self.config['model_dir'], pkl_name)

    def _get_stats_data(self, years, gid, code):
        """ Grab baseline, baseline_adjusted, and surfrad for stats """
        df_base = None
        df_base_adj = None
        df_surf = None

        for year in years:
            tmp_base = self._get_baseline_df(self.config['fp_baseline'],
                                             gid, year)
            tmp_base_adj = self._get_baseline_df(self.config[
                'fp_baseline_adj'], gid, year)
            tmp_surf = self._get_surfrad_df(self.config['fp_surf'],
                                            code, year)
            if df_base is None:
                df_base = tmp_base
                df_base_adj = tmp_base_adj
                df_surf = tmp_surf
            else:
                df_base = df_base.append(tmp_base)
                df_base_adj = df_base_adj.append(tmp_base_adj)
                df_surf = df_surf.append(tmp_surf)

        return df_base, df_base_adj, df_surf

    @staticmethod
    def _get_baseline_df(fp_baseline, gid, year):
        fname = fp_baseline.format(year=year, yy=year % 100)
        logger.debug('Getting gid {} from {}'.format(gid, fname))
        with MultiFileResource(fname) as res:
            df = pd.DataFrame({'ghi': res['ghi', :, gid],
                               'dni': res['dni', :, gid],
                               'cloud_type': res['cloud_type', :, gid],
                               'fill_flag': res['fill_flag', :, gid],
                               'solar_zenith_angle': res['solar_zenith_angle',
                                                         :, gid],
                               }, index=res.time_index)
        return df

    @staticmethod
    def _get_surfrad_df(fp_surf, code, year, window=15):
        """ @param code: three digit surfrad site code """
        logger.debug('Getting surfrad data from {}'
                     ''.format(os.path.basename(fp_surf.format(code=code,
                                                               year=year))))
        with Surfrad(fp_surf.format(code=code, year=year)) as surf:
            df_surf = surf.get_df(dt_out='5min', window_minutes=window)
        return df_surf

    @staticmethod
    def _get_all_sky_args(gid, all_sky_vars, df_all_sky):
        args = {}

        for var in all_sky_vars:
            gid_mask = (df_all_sky.gid == gid)
            arr = df_all_sky.loc[gid_mask, var].values
            if len(arr.shape) == 1:
                arr = np.expand_dims(arr, axis=1)
            args[var] = arr

        args['time_index'] = df_all_sky.loc[gid_mask,
                                            'time_index'].values.flatten()
        return args


class AutoXVal:
    """
    Run cross validation by both varying the number of sites used for
    training, and the site used for validation.
    """
    def __init__(self, sites=[0, 1, 2, 3, 4, 5, 6], val_sites=None,
                 years=(2018,), config=CONFIG, shuffle_train=False,
                 seed=None, xval=XVal, catch_nan=False, min_train=1,
                 write_stats=True):
        """
        Parameters
        ----------
        sites: list
            Sites to use for training and validation
        val_sites: None | int | list
            Site(s) to use for validation, use all if None
        years: tuple
            Years of data to use
        config: dict
            Dict of XVal configuration options. See CONFIG for example.
        shuffle_train: bool
            Randomize training site list before iterating over # of training
            sites.
        seed: None | int
            Seed for numpy.random if int
        XVal: Class
            Cross validation class. Used for testing. TODO - remove?
        catch_nan: bool
            If true, catch loss=nan exceptions and continue analysis
        min_train: int
            Minimum # of sites to use for training
        write_stats: bool
            Write statistics to disc if True
        """
        if seed is not None:
            np.random.seed(seed)

        if val_sites is None:
            val_sites = sites
        elif isinstance(val_sites, int):
            val_sites = [val_sites]
        elif isinstance(val_sites, str):
            val_sites = [int(val_sites)]

        logger.info('AXV: training sites are {}, val sites are {}'
                    ''.format(sites, val_sites))

        stats = None
        val_data = ValidationData(years, config)

        for val_site in val_sites:
            all_train_sites = [x for x in sites if x != val_site]
            if shuffle_train:
                np.random.shuffle(all_train_sites)

            logger.info('AXV: for val {}, training on {}'
                        ''.format(val_site, all_train_sites))

            for i in range(min_train - 1, len(all_train_sites)):
                train_sites = all_train_sites[0:i + 1]

                try:
                    xv = xval(train_sites=train_sites, val_site=val_site,
                              config=config, years=years, val_data=val_data)
                except ArithmeticError as e:
                    if catch_nan:
                        logger.warning('Loss=nan, val on {}, train on {}'
                                       ''.format(val_site, train_sites))
                        continue
                    else:
                        raise e

                # The _ prevents the 0 from being trimmed off
                ts = '_' + ''.join([str(x) for x in train_sites])
                xv.stats['val_site'] = val_site
                xv.stats['train_sites'] = ts
                xv.stats['num_ts'] = len(ts) - 1

                if stats is None:
                    stats = xv.stats
                else:
                    stats = pd.concat([stats, xv.stats])

            self.stats = stats.reset_index()

        if write_stats:
            train_name = ''.join([str(x) for x in all_train_sites])
            self.stats.to_csv('axv_stats_{}_{}.csv'
                              ''.format(train_name, val_sites))


class ValidationData:
    """ Load and prep validation data """
    def __init__(self, years, config):
        self.years = years
        self.config = config
        self.load_data()
        self.prep_data()

    def load_data(self):
        """ Load validation data """
        logger.debug('Loading validation data')
        surf_meta = pd.read_csv(self.config['fp'], index_col=0)
        surf_meta.index.name = 'gid'
        self.surf_meta = surf_meta[['surfrad_id']]

        df_raw = None
        df_all_sky = None

        all_gids = self.surf_meta.index.unique().values.tolist()
        for year in self.years:
            logger.debug('Loading data for {}, {}, {}'
                         ''.format(year, all_gids, self.config['features']))
            with NSRDBFeatures(self.config[
                    'fp_data'].format(year=year)) as res:
                temp_raw = res.extract_features(all_gids,
                                                self.config['features'])
                temp_all_sky = res.extract_features(all_gids,
                                                    self.config[
                                                        'all_sky_vars'])
                if df_raw is None:
                    df_raw = temp_raw
                    df_all_sky = temp_all_sky
                else:
                    df_raw = df_raw.append(temp_raw)
                    df_all_sky = df_all_sky.append(temp_all_sky)

        df_raw.reset_index(drop=True, inplace=True)
        df_all_sky.reset_index(drop=True, inplace=True)

        self.df_feature_val = clean_cloud_df(df_raw, filter_daylight=False,
                                             filter_clear=False,
                                             add_feature_flag=True, sza_lim=89)
        self.df_all_sky_val = clean_cloud_df(df_all_sky, filter_daylight=False,
                                             filter_clear=False,
                                             add_feature_flag=True, sza_lim=89)

    def prep_data(self):
        """ Prepare validation data """
        logger.debug('Prepping validation data')

        day_mask = (self.df_feature_val['solar_zenith_angle'] < 89)
        cloud_mask = (day_mask & self.df_feature_val['cloud_type']
                      .isin(ICE_TYPES + WATER_TYPES))

        # TODO - handle the masks better
        self.mask = cloud_mask  # let phygnn predict only cloudsse
        self.mask = day_mask  # let phygnn predict clearsky
        features = [c for c in self.df_feature_val.columns if
                    c not in self.config['ignore']]
        proc = PreProcess(self.df_feature_val.loc[self.mask, features])
        self.df_x_val = proc.process_one_hot()

        # TODO: is this needed?
        features = [c for c in self.df_x_val.columns if c not in
                    self.config['ignore']]


class TrainData:
    """ Load and prep training data """
    def __init__(self, years, train_sites, config):
        self.years = years
        self.train_sites = train_sites
        self.config = config
        logger.info('Loading training data')
        self.load_data()
        logger.info('Prepping training data')
        self.prep_data()

    def load_data(self):
        """ Load training data """
        surf_meta = pd.read_csv(self.config['fp'], index_col=0)
        surf_meta.index.name = 'gid'
        self.surf_meta = surf_meta[['surfrad_id']]

        df_raw = None
        df_all_sky = None

        # ------ Grab NSRDB data for weather properties
        for year in self.years:
            logger.debug('Loading data for {}, {}, {}'
                         ''.format(year, self.train_sites,
                                   self.config['features']))
            with NSRDBFeatures(self.config[
                    'fp_data'].format(year=year)) as res:
                temp_raw = res.extract_features(self.train_sites,
                                                self.config['features'])
                temp_all_sky = res.extract_features(self.train_sites,
                                                    self.config[
                                                        'all_sky_vars'])
                if df_raw is None:
                    df_raw = temp_raw
                    df_all_sky = temp_all_sky
                else:
                    df_raw = df_raw.append(temp_raw, ignore_index=True)
                    df_all_sky = df_all_sky.append(temp_all_sky,
                                                   ignore_index=True)

        logger.debug('Shape df_raw={}, df_all_sky={}'.format(df_raw.shape,
                                                             df_all_sky.shape))

        logger.debug('Converting to 2D for all_sky')
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

        # ------Grab surface data
        df_surf = None
        for year in self.years:
            logger.debug('Grabbing surface data for {} and {}'
                         ''.format(year, self.train_sites))
            for gid in self.train_sites:
                code = surf_meta.loc[gid, 'surfrad_id']
                with Surfrad(self.config['fp_surf'].format(year=year,
                                                           code=code)) as surf:
                    temp = surf.get_df(dt_out='5min', window_minutes=15)
                temp['gid'] = gid
                temp['time_index'] = temp.index.values
                if df_surf is None:
                    df_surf = temp
                else:
                    df_surf = df_surf.append(temp, ignore_index=True)
        logger.debug('Shape: df_surf={}'.format(df_surf.shape))

        assert all(df_all_sky.gid.values == df_surf.gid.values)
        assert all(df_all_sky.time_index.values == df_surf.time_index.values)

        df_surf = df_surf.reset_index(drop=True)
        df_surf = df_surf.drop(['gid', 'time_index'], axis=1)
        df_all_sky = df_all_sky.join(df_surf)

        self.df_raw = df_raw
        self.df_all_sky = df_all_sky.interpolate('nearest').bfill().ffill()
        self.p_kwargs = {'labels': df_all_sky.columns.values.tolist()}

    def prep_data(self, filter_clear=False):
        """ Clean and prepare training data """
        logger.debug('Cleaning df_raw')
        logger.debug('Shape: df_raw={}'.format(self.df_raw.shape))
        self.df_train = clean_cloud_df(self.df_raw, filter_clear=filter_clear)
        logger.debug('Shape: df_train={}'.format(self.df_train.shape))
        logger.debug('Cleaning df_all_sky')
        self.df_all_sky = clean_cloud_df(self.df_all_sky,
                                         filter_clear=filter_clear)
        # Inspecting features would go here

        # Final cleaning and one-hot encoding
        for name in self.config['drop_list']:
            if name in self.df_train:
                self.df_train = self.df_train.drop(name, axis=1)
        logger.debug('*Shape: df_train={}'.format(self.df_train.shape))
        proc = PreProcess(self.df_train)
        self.df_train = proc.process_one_hot()
        logger.debug('**Shape: df_train={}'.format(self.df_train.shape))
        features = self.df_train.columns.values.tolist()

        features = [f for f in features if f not in self.config['ignore']]

        self.y = self.df_train[self.config['y_labels']].values
        self.x = self.df_train[features].values
        self.p = self.df_all_sky.values

        logger.debug('Shapes: x={}, y={}, p={}'.format(self.x.shape,
                                                       self.y.shape,
                                                       self.p.shape))
        assert self.y.shape[0] == self.x.shape[0] == self.p.shape[0]


def clean_cloud_df(cloud_df_raw, filter_daylight=True, filter_clear=True,
                   add_feature_flag=True, sza_lim=89):
    """ Clean up cloud data """
    t0 = time.time()
    cloud_df = cloud_df_raw.copy()
    day = (cloud_df['solar_zenith_angle'] < sza_lim)

    day_missing_ctype = day & (cloud_df['cloud_type'] < 0)
    cloud_df.loc[(cloud_df['cloud_type'] < 0), 'cloud_type'] = np.nan
    cloud_df['cloud_type'] = cloud_df['cloud_type']\
        .interpolate('nearest').ffill().bfill()

    cloudy = cloud_df['cloud_type'].isin(ICE_TYPES + WATER_TYPES)
    day_clouds = day & cloudy
    day_missing_opd = day_clouds & (cloud_df['cld_opd_dcomp'] <= 0)
    day_missing_reff = day_clouds & (cloud_df['cld_reff_dcomp'] <= 0)
    cloud_df.loc[(cloud_df['cld_opd_dcomp'] <= 0), 'cld_opd_dcomp'] = np.nan
    cloud_df.loc[(cloud_df['cld_reff_dcomp'] <= 0), 'cld_reff_dcomp'] = np.nan

    logger.debug('{:.2f}% of timesteps are daylight'
                 .format(100 * day.sum() / len(day)))
    logger.debug('{:.2f}% of daylight timesteps are cloudy'
                 .format(100 * day_clouds.sum() / day.sum()))
    logger.debug('{:.2f}% of daylight timesteps are missing cloud type'
                 .format(100 * day_missing_ctype.sum() / day.sum()))
    logger.debug('{:.2f}% of cloudy daylight timesteps are missing cloud opd'
                 .format(100 * day_missing_opd.sum() / day_clouds.sum()))
    logger.debug('{:.2f}% of cloudy daylight timesteps are missing cloud reff'
                 .format(100 * day_missing_reff.sum() / day_clouds.sum()))

    cloud_df = cloud_df.interpolate('nearest').ffill().bfill()
    cloud_df.loc[~cloudy, 'cld_opd_dcomp'] = 0.0
    cloud_df.loc[~cloudy, 'cld_reff_dcomp'] = 0.0

    assert ~any(cloud_df['cloud_type'] < 0)
    assert ~any(pd.isna(cloud_df))
    assert ~any(cloudy & (cloud_df['cld_opd_dcomp'] <= 0))

    if add_feature_flag:
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
        mask = mask & day

    if filter_clear:
        mask = mask & cloudy

    if filter_daylight or filter_clear:
        logger.debug('Data reduced from '
                     '{} rows to {} after filters ({:.2f}% of original)'
                     .format(len(cloud_df), mask.sum(),
                             100 * mask.sum() / len(cloud_df)))
        cloud_df = cloud_df[mask]

    logger.debug('Cleaning took {:.1f} seconds'.format(time.time() - t0))

    return cloud_df
