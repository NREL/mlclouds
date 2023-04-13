"""
mlclouds phygnn model trainer and validator classes.
"""

import logging
import numpy as np
import pandas as pd
import os

from rex.multi_file_resource import MultiFileResource

from nsrdb.file_handlers.surfrad import Surfrad
from nsrdb.all_sky.all_sky import all_sky
from nsrdb.utilities.statistics import mae_perc, mbe_perc, rmse_perc

from farms import ICE_TYPES, WATER_TYPES, CLEAR_TYPES

from mlclouds.data_handlers import ValidationData
from mlclouds.utilities import FP_BASELINE, ALL_SKY_VARS
from mlclouds.utilities import FP_SURFRAD_DATA, CONFIG
from mlclouds.utilities import surf_meta, calc_time_step

logger = logging.getLogger(__name__)


class Validator:
    """
    Run PhygnnModel predictions, run allsky using predicted cloud properties
    and compare to NSRDB baseline irradiance.
    """
    def __init__(self, model, config=CONFIG, val_files=None, val_data=None,
                 update_clear=False, update_cloudy=False, test_set_mask=None,
                 save_timeseries=False):
        """
        Parameters
        ----------
        model: Phygnn instance
            Trained or loaded Phygnn model
        config: dict
            Phygnn configuration dict
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
        test_set_mask: None | numpy.ndarray of bool
            Set of full data set in val_files to use. If None, use full
            dataset.
        save_timeseries: bool
            Save time series data to disk
        """
        if (val_files is None and val_data is None) or \
           (val_files is not None and val_data is not None):
            msg = 'Either val_files or val_data must be set, but not both'
            logger.error(msg)
            raise ValueError(msg)

        self._config = config

        if not isinstance(val_files, list):
            val_files = [val_files]

        if val_data is None:
            all_sky_vars = self._config.get('all_sky_vars', ALL_SKY_VARS)
            vd = ValidationData(val_files=val_files,
                                features=config['features'],
                                y_labels=config['y_labels'],
                                all_sky_vars=all_sky_vars,
                                one_hot_cats=config['one_hot_categories'],
                                test_set_mask=test_set_mask,
                                )
            val_data = vd

        self.df_x_val = val_data.df_x_val
        self.df_all_sky_val = val_data.df_all_sky_val
        self.df_feature_val = val_data.df_feature_val
        self.mask = val_data.mask
        self.files_meta = val_data.files_meta
        self.val_data = val_data

        self._predict(model, update_clear, update_cloudy)
        self._calc_stats(test_set_mask, save_timeseries=save_timeseries)

    def _predict(self, model, update_clear, update_cloudy):
        """
        Use PHYGNN model to predict cloud properties.

        Parameters
        ----------
        model: Phygnn instance
            Trained or loaded Phygnn model
        update_clear: bool
            If true, update cloud type for clear time steps with phygnn
            predictions
        update_cloudy: bool
            If true, update cloud type for cloudy time steps with phygnn
            predictions
        """
        logger.info('Predicting opd and reff')
        predicted_raw = model.predict(self.df_x_val)
        assert not predicted_raw.isnull().values.any()
        logger.debug('Predicted data shape is {}'.format(predicted_raw.shape))

        # TODO - use config['y_labels'] for this
        opd_raw = predicted_raw['cld_opd_dcomp'].values
        reff_raw = predicted_raw['cld_reff_dcomp'].values

        opd = np.minimum(opd_raw, 160)
        reff = np.minimum(reff_raw, 160)
        opd = np.maximum(opd, 0.0)
        reff = np.maximum(reff, 0.0)

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
            logger.debug('The Phygnn predicted {} additional clear timesteps'
                         '({:.2f}%)'.format(mask.sum(),
                                            100 * mask.sum() / len(mask)))
        if update_cloudy:
            mask = ((self.df_feature_val['solar_zenith_angle'] < 89)
                    & (self.df_feature_val['cld_opd_dcomp'] > 0.0)
                    & (self.df_feature_val['cloud_type'].isin(CLEAR_TYPES)))
            self.df_feature_val.loc[mask, 'cloud_type'] = 8
            self.df_all_sky_val.loc[mask, 'cloud_type'] = 8
            logger.debug('The Phygnn predicted {} additional cloudy timesteps '
                         '({:.2f}%)'.format(mask.sum(),
                                            100 * mask.sum() / len(mask)))

        clear_mask = self.df_feature_val['cloud_type'].isin(CLEAR_TYPES)
        self.df_feature_val.loc[clear_mask, 'cld_opd_dcomp'] = 0
        self.df_feature_val.loc[clear_mask, 'cld_reff_dcomp'] = 0
        self.df_all_sky_val.loc[clear_mask, 'cld_opd_dcomp'] = 0
        self.df_all_sky_val.loc[clear_mask, 'cld_reff_dcomp'] = 0
        # end TODO

        logger.debug('shapes: df_feature_val={}, df_all_sky_val={}'
                     ''.format(self.df_feature_val.shape,
                               self.df_all_sky_val.shape))

        assert not self.df_feature_val.isnull().values.any()
        assert not self.df_all_sky_val.isnull().values.any()

    def _calc_stats(self, test_set_mask, save_timeseries=False):
        """
        Calculate accuracy of PHYGNN model predictions

        Parameters
        ----------
        test_set_mask: None | numpy.ndarray of bool
            Set of full data set in val_files to use. If None, use full
            dataset.
        save_timeseries: bool
            Save time series data to disk
        """
        fp_baseline = FP_BASELINE

        logger.info('Calculating statistics')

        gids = [k for k, v in surf_meta().to_dict()['surfrad_id'].items()]
        logger.debug('Calcing stats for gids: {}'.format(gids))

        s_data = self._get_stats_data(self.files_meta, gids, fp_baseline)
        df_base_full, df_surf_full = s_data

        logger.debug('Shapes: df_base_full={}, '
                     'df_surf_full={}'.format(df_base_full.shape,
                                              df_surf_full.shape))

        if test_set_mask is not None:
            df_base_full = df_base_full[test_set_mask]
            df_surf_full = df_surf_full[test_set_mask]
            logger.debug('Test set shapes: df_base_full={}, '
                         'df_surf_full={}'.format(df_base_full.shape,
                                                  df_surf_full.shape))

        i = 0
        stats = pd.DataFrame(columns=['Model', 'Site', 'Variable',
                                      'Condition', 'N'])
        for gid in gids:
            code = surf_meta().loc[gid, 'surfrad_id']
            logger.debug('Computing stats for gid: {} {}'.format(gid, code))

            idx = df_base_full.gid == gid
            df_baseline = df_base_full[idx]
            df_surf = df_surf_full[idx]

            logger.debug('Shapes: df_baseline={}, '
                         'df_surf={}'.format(df_baseline.shape,
                                             df_surf.shape))

            # Run all_sky for current gid
            all_sky_vars = self._config.get('all_sky_vars', ALL_SKY_VARS)
            args = self._get_all_sky_args(gid, self.df_all_sky_val,
                                          all_sky_vars=all_sky_vars)
            out = all_sky(**args)
            index = pd.DatetimeIndex(args['time_index']).tz_localize('utc')
            all_sky_out = pd.DataFrame({k: v.flatten() for k, v in
                                        out.items()}, index=index)

            gid_mask = (self.df_all_sky_val.gid == gid)
            val_daylight_mask = (self.df_all_sky_val.loc[gid_mask,
                                 'solar_zenith_angle'] < 89).values
            val_cloudy_mask = (val_daylight_mask
                               & (self.df_all_sky_val
                                  .loc[gid_mask, 'cloud_type']
                                  .isin(ICE_TYPES + WATER_TYPES)).values)
            val_clear_mask = (val_daylight_mask
                              & (self.df_all_sky_val
                                 .loc[gid_mask, 'cloud_type']
                                 .isin(CLEAR_TYPES)).values)
            val_bad_cloud_mask = (val_cloudy_mask
                                  & (self.df_all_sky_val
                                     .loc[gid_mask, 'flag'] == 'bad_cloud')
                                  .values)

            m_iter = [[val_daylight_mask, 'All-Sky'],
                      [val_cloudy_mask, 'Cloudy'],
                      [val_clear_mask, 'Clear'],
                      [val_bad_cloud_mask, 'Missing Cloud Data']]

            if save_timeseries:
                for var in ['dni', 'ghi']:
                    self._timeseries_to_csv(gid, var,
                                            df_baseline[var],
                                            all_sky_out[var],
                                            df_surf['surfrad_' + var])

            for mask, condition in m_iter:
                for var in ['dni', 'ghi']:
                    baseline = df_baseline.loc[mask, var].values
                    surf = df_surf.loc[mask, 'surfrad_' + var].values
                    mlclouds = all_sky_out.loc[mask, var].values

                    mae_baseline = mae_perc(baseline, surf)
                    mbe_baseline = mbe_perc(baseline, surf)
                    rmse_baseline = rmse_perc(baseline, surf)

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
                    stats.at[i, 'N'] = mask.sum()
                    i += 1

                    stats.at[i, 'Model'] = 'MLClouds'
                    stats.at[i, 'Site'] = code.upper()
                    stats.at[i, 'Variable'] = var.upper()
                    stats.at[i, 'Condition'] = condition
                    stats.at[i, 'MAE (%)'] = mae_ml
                    stats.at[i, 'MBE (%)'] = mbe_ml
                    stats.at[i, 'RMSE (%)'] = rmse_ml
                    stats.at[i, 'N'] = mask.sum()
                    i += 1

        self.stats = stats
        logger.info('Finished computing stats.')

    def _timeseries_to_csv(self, gid, var, baseline, mlclouds, surf):
        """
        Save irradiance timseries data to disk for later analysis

        Parameters
        ----------
        gid: int
            Gid of site excluded from training, only used for file naming
        var: str
            Irradiance type of data: dni or ghi
        baseline: pd.Series
            Baseline NSRDB irradiance
        mlclouds: pd.Series
            Irradiance as predicted by PHYGNN
        surf: pd.Series
            Ground measured irradiance
        """
        df = pd.DataFrame({'Baseline': baseline,
                           'PHYGNN': mlclouds,
                           'Surfrad': surf})
        tdir = self._config.get('timeseries_dir', 'timeseries/')
        if not os.path.exists(tdir):
            os.makedirs(tdir)
        df.to_csv(os.path.join(tdir, 'timeseries_{}_{}.csv'.format(var, gid)))

    def _get_stats_data(self, files_meta, gids, fp_baseline):
        """
        Grab baseline and surfrad for stats. Adjust time
        steps as necessary to match validation satellite data.

        Parameters
        ----------
        files_meta: dict
            Year, time step, and time_index for all satellite validation files
        gids: int
            Gids of desired surfrad site
        fp_baseline: str
            Full path of NSRDB baseline irradiance

        Returns
        -------
        df_base: pd.Dataframe
            NSRDB irradiance
        df_surf: pd.Dataframe
            Ground measured irradiance
        """
        df_base = None
        df_surf = None
        for year_meta in files_meta:
            year = year_meta['year']
            area = year_meta['area']

            logger.debug('Loading data for {} / {}'.format(year, area))
            for gid in gids:
                tmp_base = self._get_baseline_df(fp_baseline, gid, year, area)
                tmp_surf = self._get_surfrad_df(gid, year,
                                                tstep=year_meta['time_step'])

                if df_base is None:
                    df_base = tmp_base
                    df_surf = tmp_surf
                else:
                    df_base = pd.concat([df_base, tmp_base])
                    df_surf = pd.concat([df_surf, tmp_surf])

        assert (df_base.gid == df_surf.gid).all()
        return df_base, df_surf

    def _get_surfrad_df(self, gid, year, tstep=5, window_default=15):
        """
        Get ground measured irradiance data for location and year

        Parameters
        ----------
        gid: int
            Gid of desired surfrad site
        year: int
            Four digit year
        tstep: int
            timestep size for output (minutes)
        window_default: int
            Minutes that the moving average window will be over

        Returns
        -------
        df_surf: pandas.DataFrame
            Surfrad data
        """
        code = surf_meta().loc[gid, 'surfrad_id']
        fp_surf = FP_SURFRAD_DATA.format(code=code, year=year)
        logger.debug('\tGetting surfrad data for {} from {}'
                     ''.format(gid, os.path.basename(fp_surf)))

        w = self._config.get('surfrad_window_minutes', window_default)
        with Surfrad(fp_surf) as surf:
            df_surf = surf.get_df(dt_out='{}min'.format(tstep),
                                  window_minutes=w)

        # add prefix to avoid confusion
        df_surf = df_surf.rename({'ghi': 'surfrad_ghi',
                                  'dni': 'surfrad_dni',
                                  'dhi': 'surfrad_dhi'}, axis=1)
        df_surf['gid'] = gid
        return df_surf

    @staticmethod
    def _get_baseline_df(fp_baseline, gid, year, area):
        """
        Get baseline NSRDB irradiance for surfrad site, area, and satellite

        Parameters
        ----------
        fp_baseline: str
            Full path to h5 file with data
        gid: int
            Desired surfrad site
        year: int
            Desired year
        area: str
            Desired satellite, east or west

        Returns
        -------
        df : pandas.DataFrame
            Data frame with ghi, dni, cloud_type, fill_flag, and sza
        """
        fname = fp_baseline.format(year=year, yy=year % 100, area=area)
        logger.debug('\tGetting gid {} from {}'
                     ''.format(gid, os.path.basename(fname)))
        with MultiFileResource(fname) as res:
            df = pd.DataFrame({'ghi': res['ghi', :, gid],
                               'dni': res['dni', :, gid],
                               'cloud_type': res['cloud_type', :, gid],
                               'fill_flag': res['fill_flag', :, gid],
                               'solar_zenith_angle': res['solar_zenith_angle',
                                                         :, gid],
                               }, index=res.time_index)
        df['gid'] = gid
        return df

    @staticmethod
    def _get_all_sky_args(gid, df_all_sky, all_sky_vars=ALL_SKY_VARS):
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
