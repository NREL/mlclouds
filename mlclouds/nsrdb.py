"""NSRDB feature extractor"""

import logging

import numpy as np
import pandas as pd
from farms import CLEAR_TYPES, ICE_TYPES, WATER_TYPES
from rex import NSRDB

logger = logging.getLogger(__name__)


class NSRDBFeatures(NSRDB):
    """NSRDB Features extractor"""

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
            if f in ("clear_fraction", "ice_fraction", "water_fraction"):
                f_data = self["cloud_type", :, sites].astype(np.float32)
            else:
                f_data = self[f, :, sites]
            if isinstance(sites, slice):
                step = sites.step if sites.step is not None else 0
                sites = list(range(sites.start, sites.stop, step))

            f_data = pd.DataFrame(f_data, index=self.time_index, columns=sites)
            columns = {"level_0": "gid", "level_1": "time_index", 0: f}
            f_data = f_data.unstack().reset_index().rename(columns=columns)
            if features_df is None:
                features_df = f_data
            else:
                features_df = pd.merge(
                    features_df, f_data, on=["gid", "time_index"]
                )

        if "water_fraction" in features_df:
            mask = features_df["water_fraction"].isin(WATER_TYPES)
            features_df.loc[mask, "water_fraction"] = 1
            features_df.loc[~mask, "water_fraction"] = 0
        if "ice_fraction" in features_df:
            mask = features_df["ice_fraction"].isin(ICE_TYPES)
            features_df.loc[mask, "ice_fraction"] = 1
            features_df.loc[~mask, "ice_fraction"] = 0
        if "clear_fraction" in features_df:
            mask = features_df["clear_fraction"].isin(CLEAR_TYPES)
            features_df.loc[mask, "clear_fraction"] = 1
            features_df.loc[~mask, "clear_fraction"] = 0
        return features_df
