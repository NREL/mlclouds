"""Wrapped phygnn model with methods to produce needed outputs for NSRDB."""

import numpy as np
import pandas as pd
from phygnn import PhygnnModel

from mlclouds.p_fun import _get_sky_type


class MLCloudsModel(PhygnnModel):
    """Extended phygnn model with methods for interfacing with NSRDB."""

    CTYPE_FRACTIONS = ('clear_fraction', 'ice_fraction', 'water_fraction')

    @property
    def predicts_cloud_fractions(self):
        """Check if this model predicts cloud type fractions."""
        return all(f in self.label_names for f in self.CTYPE_FRACTIONS)

    def predict(self, *args, **kwargs):
        """Convert cloud type fractions into a integer cloud type and remove
        cloud type fractions from output, if cloud type fractions are predicted
        by this model. Otherwise, return output without any additional
        processing."""
        out = super().predict(*args, **kwargs)
        is_array = not hasattr(out, 'columns')
        if self.predicts_cloud_fractions:
            if is_array:
                out = pd.DataFrame(columns=self.label_names, data=out)
            fracs = {f: out[f].values for f in self.CTYPE_FRACTIONS}
            out['cloud_type'] = _get_sky_type(**fracs)
            out = out[self.output_names]
        return out if not is_array else np.asarray(out)

    @property
    def output_names(self):
        """Remove cloud type fraction labels from features and replace with
        "cloud_type", if this model predicts cloud type fractions. Otherwise,
        just return labels unchanged."""
        output_names = self.label_names.copy()
        if self.predicts_cloud_fractions:
            output_names = [
                f for f in output_names if f not in self.CTYPE_FRACTIONS
            ]
            output_names += ['cloud_type']
        return output_names
