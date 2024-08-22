"""Wrapped phygnn model with methods to produce needed outputs for NSRDB."""

import numpy as np
import pandas as pd
from phygnn import PhygnnModel

from mlclouds.p_fun import _get_sky_type


class MLCloudsModel(PhygnnModel):
    """Extended phygnn model with methods for interfacing with NSRDB."""

    def predict(self, *args, **kwargs):
        """Override predict method to return cloud type if cloud type fractions
        are predicted by this model."""
        out = super().predict(*args, **kwargs)
        frac_names = ("clear_fraction", "ice_fraction", "water_fraction")
        is_array = not hasattr(out, "columns")
        if all(f in self.label_names for f in frac_names):
            if is_array:
                out = pd.DataFrame(columns=self.label_names, data=out)
            fracs = {f: out[f].values for f in frac_names}
            out["cloud_type"] = _get_sky_type(**fracs)
            out_feats = [f for f in self.label_names if f not in frac_names]
            out_feats += ["cloud_type"]
            out = out[out_feats]
        return out if not is_array else np.asarray(out)

    @property
    def output_names(self):
        """Output feature names with parsing of cloud type fractions if the
        model predicts cloud types."""
        frac_names = ("clear_fraction", "ice_fraction", "water_fraction")
        output_names = self.label_names
        if all(f in output_names for f in frac_names):
            output_names = [f for f in output_names if f not in frac_names]
            output_names += ["cloud_type"]
        return output_names
