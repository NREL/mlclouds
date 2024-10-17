"""Wrapped phygnn model with methods to produce needed outputs for NSRDB."""

import numpy as np
import pandas as pd
from phygnn import PhygnnModel

from mlclouds.p_fun import _get_sky_type


class MLCloudsModel(PhygnnModel):
    """Extended phygnn model with methods for interfacing with NSRDB. This
    includes conversion of cloud type fraction predictions into a single
    integer ``cloud_type`` feature, if the model predicts cloud type fractions
    in addition to the standard cloud property predictions"""

    CTYPE_FRACTIONS = ('clear_fraction', 'ice_fraction', 'water_fraction')

    @property
    def predicts_cloud_fractions(self):
        """Check if this model predicts cloud type fractions."""
        return all(f in self.label_names for f in self.CTYPE_FRACTIONS)

    def predict(self, *args, **kwargs):
        """First, use model to predict label from given features. Then, if
        clould type fractions are predicted by this model, convert these
        fractions into a integer cloud type and remove cloud type fractions
        from output. Otherwise, return output without any additional
        processing.

        Parameters
        ----------
        *args : list
            List of positional arguments for ``PhygnnModel``
        **kwargs: Mapping
            Mappable of keyword arguments for ``PhygnnModel``

        Returns
        -------
        prediction : ndarray | pandas.DataFrame
            label prediction
        """
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
        """Ordered list of predicted features. If this model predicts cloud
        type fractions then these are replaced with a single ``cloud_type``
        output name. This matches the conversion from cloud type fractions to
        ``cloud_type`` in :meth:`predict`.

        Returns
        -------
        output_names : list
            Ordered list of predicted features. If the model predicts cloud
            type fractions these are replaced with ``cloud_type``
        """
        output_names = self.label_names.copy()
        if self.predicts_cloud_fractions:
            output_names = [
                f for f in output_names if f not in self.CTYPE_FRACTIONS
            ]
            output_names += ['cloud_type']
        return output_names
