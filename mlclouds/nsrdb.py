# -*- coding: utf-8 -*-
"""
NSRDB feature extractor
"""
import logging
import pandas as pd
from rex import NSRDB

logger = logging.getLogger(__name__)


class NSRDBFeatures(NSRDB):
    """
    NSRDB Features extractor
    """

    def extract_features(self, sites, features):
        """
        Extract features from given sites and return as a dataframe

        Parameters
        ----------
        sites : int | list | slice
            sites to extract from NSRDB
        features : str | list
            Dataset(s) to extract from NSRDB

        Returns
        -------
        features_df : pandas.DataFrame
            Features DataFrame
        """
        if isinstance(features, str):
            features = [features]
        elif isinstance(features, tuple):
            features = list(features)

        features_df = None
        for f in features:
            f_data = self[f, :, sites]
            if isinstance(sites, slice):
                step = sites.step if sites.step is not None else 0
                sites = list(range(sites.start, sites.stop, step))

            f_data = pd.DataFrame(f_data, index=self.time_index,
                                  columns=sites)
            columns = {'level_0': 'gid', 'level_1': 'time_index', 0: f}
            f_data = f_data.unstack().reset_index().rename(columns=columns)
            if features_df is None:
                features_df = f_data
            else:
                features_df = pd.merge(features_df, f_data,
                                       on=['gid', 'time_index'])

        return features_df
