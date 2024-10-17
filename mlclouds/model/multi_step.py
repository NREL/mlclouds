"""Wrapped phygnn model with methods to produce needed outputs for NSRDB."""

import numpy as np

from mlclouds.p_fun import decode_sky_type, encode_sky_type

from .base import MLCloudsModel


class MultiCloudsModel(MLCloudsModel):
    """Composite MLCloudsModel with a model that predicts cloud type and a
    second model which uses the predicted cloud type to predict cloud
    properties."""

    def __init__(self, cloud_prop_model, cloud_type_model=None):
        self.cloud_type_model = cloud_type_model
        self._model = cloud_prop_model
        self._label_names = self._model._label_names
        self._norm_params = self._model._norm_params
        self._normalize = self._model._normalize

        if self.cloud_type_model is not None:
            self._feature_names = self.cloud_type_model._feature_names
            self._one_hot_categories = (
                self.cloud_type_model._one_hot_categories
            )
        else:
            self._feature_names = self._model._feature_names
            self._one_hot_categories = self._model._one_hot_categories

    @classmethod
    def load(cls, cloud_prop_model_path, cloud_type_model_path=None):
        """Load cloud property model path and optionally a cloud type model
        path."""
        cprop_model = MLCloudsModel.load(cloud_prop_model_path)
        ctype_model = (
            None
            if cloud_type_model_path is None
            else MLCloudsModel.load(cloud_type_model_path)
        )
        return cls(cloud_prop_model=cprop_model, cloud_type_model=ctype_model)

    def predict(self, features, **kwargs):
        """First, predict cloud type if this is a composite model with a cloud
        type model. Use these as the cloud type input feature for the cloud
        property model. If there is no cloud type model just use the cloud
        property model directly.

        Parameters
        ----------
        features : dict | pd.DataFrame
            Input features used for cloud property / cloud type predictions
        **kwargs: Mapping
            Mappable of keyword arguments for ``PhygnnModel``

        Returns
        -------
        prediction : ndarray | pandas.DataFrame
            label prediction
        """
        if self.cloud_type_model is not None:
            out = self.cloud_type_model.predict(features, **kwargs)
            features['flag'] = decode_sky_type(out)
        out = self.model.predict(features, **kwargs)

        if 'flag' in features:
            ctype = encode_sky_type(features)
            if hasattr(out, 'columns'):
                out['cloud_type'] = ctype
            else:
                out = np.concatenate([out, ctype[:, None]], axis=-1)
        return out

    @property
    def output_names(self):
        """Ordered list of predicted features. We include cloud_type which is
        predicted if this is a composite model with a cloud type model and
        passed through from input features otherwise.

        Returns
        -------
        output_names : list
            Ordered list of predicted features. If the model predicts cloud
            type fractions these are replaced with ``cloud_type``
        """
        return [*self.label_names.copy(), 'cloud_type']
