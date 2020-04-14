# -*- coding: utf-8 -*-
"""
TensorFlow Model handler
"""
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from warnings import warn

logger = logging.getLogger(__name__)


class TfModel:
    """
    Tensorflow based Machine Learning Model
    """
    def __init__(self, features, label, model_layers=None,
                 feature_columns=None, learning_rate=0.001,
                 loss="mean_squared_error", metrics=('mae', 'mse'),
                 **kwargs):
        """
        Parameters
        ----------
        features : pandas.DataFrame
            Table of input features to train on
        label : pandas.DataFrame
            Table of label to train on
        model_layers : list, optional
            List of tensorflow layers.Dense kwargs (dictionaries)
            if None use a single linear layer, by default None
        feature_columns : list, optional
            list of feature column descriptions (dictionaries)
            if None use numeric columns with the feature_attr name,
            by default None
        learning_rate : float, optional
            tensorflow optimizer learning rate, by default 0.001
        loss : str, optional
            name of objective function, by default "mean_squared_error"
        metrics : list, optional
            List of metrics to be evaluated by the model during training and
            testing, by default ('mae', 'mse')
        kwargs : dict
            kwargs for tensorflow.keras.models.compile
        """
        self._features, self._means, self._stdevs = \
            self._normalize(self._clean_feature_names(features))
        self._label = label

        if feature_columns is None:
            feature_layer = self._generate_feature_layer(self._features)
        else:
            if len(feature_columns) < len(self._features.columns):
                msg = ("There must be at least one feature column per feature!"
                       " {} columns were supplied but there are {} features!"
                       .format(len(feature_columns),
                               len(self._features.columns)))
                logger.error(msg)
                raise ValueError

            feature_layer = self._build_feature_layer(feature_columns)

        self._model = self._build_model(feature_layer,
                                        model_layers=model_layers,
                                        learning_rate=learning_rate,
                                        loss=loss, metrics=metrics, **kwargs)
        self._history = None

    def __repr__(self):
        msg = "{}:\n{}".format(self.__class__.__name__, self.model_summary)

        return msg

    def __getitem__(self, features):
        """
        Use model to predict label from given features

        Parameters
        ----------
        features : pandas.DataFrame
            features to predict from

        Returns
        -------
        pandas.DataFrame
            Label prediction
        """
        return self.predict(features)

    @property
    def model_summary(self):
        """
        Tensorflow model summary

        Returns
        -------
        str
        """
        try:
            summary = self._model.summary()
        except ValueError:
            summary = None

        return summary

    @property
    def source_features(self):
        """
        Original features prior to normalization

        Returns
        -------
        pandas.DataFrame
        """
        return self._unnormalize(self._features, self._means, self._stdevs)

    @property
    def model_features(self):
        """
        Normalized features

        Returns
        -------
        pandas.DataFrame
        """
        return self._features

    @property
    def norm_features(self):
        """
        Normalized features

        Returns
        -------
        pandas.DataFrame
        """
        return self._features

    @property
    def label(self):
        """
        Model label (output)

        Returns
        -------
        pandas.DataFrame
        """
        return self._label

    @property
    def feature_means(self):
        """
        Feature means, used for normalization

        Returns
        -------
        pandas.DataFrame
        """
        return self._means

    @property
    def feature_stdevs(self):
        """
        Feature standard deviations, used for normalization

        Returns
        -------
        pandas.DataFrame
        """
        return self._stdevs

    @property
    def model(self):
        """
        Trained model

        Returns
        -------
        tensorflow.keras.models
        """
        return self._model

    @property
    def history(self):
        """
        Model training history

        Returns
        -------
        pandas.DataFrame
        """
        if self._history is None:
            msg = 'Model has not been trained yet!'
            logger.warning(msg)
            warn(msg)
        history = pd.DataFrame(self._history.history)
        history['epoch'] = self._history.epoch

        return history

    @staticmethod
    def _clean_feature_names(features):
        names = features.columns
        new_names = {}
        for name in names:
            new_name = name.replace(' ', '_')
            new_name = new_name.replace('*', '-x-')
            new_name = new_name.replace('+', '-plus-')
            new_name = new_name.replace('**', '-exp-')
            new_name = new_name.replace(')', '')
            new_name = new_name.replace('log(', 'log-')
            new_names[name] = new_name

        features = features.rename(columns=new_names)

        return features

    @staticmethod
    def _normalize(features, means=None, stdevs=None):
        """
        Normalize features with mean at 0 and stdev of 1.

        Parameters
        ----------
        features : pandas.DataFrame
            features table to normalize
        means : list
            List of feature means to use for normalization
        stdevs : list
            List of feature stdevs to use for normalization

        Returns
        -------
        features : pandas.DataFrame
            features normalized with mean of 0 and stdev of 1.
        means : list
            List of original mean values of the features
        stdevs : list
            List of original standard deviations of the features.
        """
        if means is None:
            means = np.nanmean(features, axis=0)

        if stdevs is None:
            stdevs = np.nanstd(features, axis=0)

        features = features - means
        features /= stdevs

        return features, means, stdevs

    @staticmethod
    def _unnormalize(features, means, stdevs):
        """
        Unnormalize features with mean at 0 and stdev of 1.

        Parameters
        ----------
        features : pandas.DataFrame
            normalized features table
        means : list
            List of feature means to used for normalization
        stdevs : list
            List of feature stdevs to used for normalization

        Returns
        -------
        features : pandas.DataFrame
            un-normalized features
        """
        features = features * stdevs
        features += means

        return features

    @staticmethod
    def _generate_feature_layer(features):
        """
        Generate feature layer from features table

        Parameters
        ----------
        features : pandas.DataFrame
            Table of model features

        Returns
        -------
        tensorflow.keras.layers.DenseFeatures
            Feature layer instance
        """
        feature_columns = []
        for name, data in features.iteritems():
            if np.issubdtype(data.dtype.name, np.number):
                f_col = feature_column.numeric_column(name)
            else:
                f_col = \
                    feature_column.categorical_column_with_hash_bucket(name)

            feature_columns.append(f_col)

        feature_layer = layers.DenseFeatures(feature_columns)

        return feature_layer

    @staticmethod
    def _build_feature_layer(feature_columns):
        """
        Build the feature layer from given feature column descriptions

        Parameters
        ----------
        feature_columns : list, optional
            list of feature column descriptions (dictionaries)
            if None use numeric columns with the feature_attr name,
            by default None

        Returns
        -------
        tensorflow.keras.layers.DenseFeatures
            Feature layer instance
        """
        tf_columns = {}
        col_map = {}  # TODO: build map to tf.feature_column functions
        # TODO: what feature_columns need to be wrapped
        indicators = [feature_column.categorical_column_with_hash_bucket,
                      feature_column.categorical_column_with_identity,
                      feature_column.categorical_column_with_vocabulary_file,
                      feature_column.categorical_column_with_vocabulary_list,
                      feature_column.crossed_column]
        for col in feature_columns:
            name = col['name']
            f_type = col_map[col['type']]
            kwargs = col.get('kwargs', {})

            if f_type == feature_column.crossed_column:
                cross_cols = [tf_columns[name]
                              for name in col['cross_columns']]
                f_col = f_type(cross_cols, **kwargs)
            elif f_type == feature_column.embedding_column:
                embedded_type = col_map[col['embedded_col']]
                f_col = embedded_type(name, **kwargs)
                f_col = f_type(f_col, **kwargs)
            else:
                f_col = f_type(name, **kwargs)

            if f_type in indicators:
                f_col = tf.feature_column.indicator_column(f_col)

            tf_columns[name] = f_col

        feature_layer = tf.keras.layers.DenseFeatures(tf_columns.values())

        return feature_layer

    @staticmethod
    def _build_model(feature_layer, model_layers=None, learning_rate=0.001,
                     loss="mean_squared_error", metrics=('mae', 'mse'),
                     **kwargs):
        """
        Build tensorflow sequential model from given layers and kwargs

        Parameters
        ----------
        feature_layer : tensorflow.keras.layers.DenseFeatures
            Feature layer instance
        model_layers : list, optional
            List of tensorflow layers.Dense kwargs (dictionaries)
            if None use a single linear layer, by default None
        learning_rate : float, optional
            tensorflow optimizer learning rate, by default 0.001
        loss : str, optional
            name of objective function, by default "mean_squared_error"
        metrics : list, optional
            List of metrics to be evaluated by the model during training and
            testing, by default ('mae', 'mse')
        kwargs : dict
            kwargs for tensorflow.keras.models.compile

        Returns
        -------
        tensorflow.keras.models.Sequential
            Compiled tensorflow Sequential model
        """
        model = tf.keras.models.Sequential()
        model.add(feature_layer)
        if model_layers is None:
            # Add a single linear layer
            model.add(layers.Dense(units=1, input_shape=(1,)))
        else:
            for layer in model_layers:
                model.add(layers.Dense(**layer))

        if isinstance(metrics, tuple):
            metrics = list(metrics)
        elif not isinstance(metrics, list):
            metrics = [metrics]

        optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                      **kwargs)

        return model

    def train(self, epochs=100, validation_split=0.2, early_stop=True,
              **kwargs):
        """
        Train the model with the provided features and label

        Parameters
        ----------
        epochs : int, optional
            Number of epochs to train the model, by default 100
        validation_split : float, optional
            Fraction of the training data to be used as validation data,
            by default 0.2
        early_stop : bool
            Flag to stop training when it stops improving
        kwargs : dict
            kwargs for tensorflow.keras.models.fit
        """
        if self._history is not None:
            msg = 'Model has already been trained and will be re-fit!'
            logger.warning(msg)
            warn(msg)

        features = {name: np.array(value)
                    for name, value in self._features.items()}
        label = self._label.values
        if early_stop:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=10)
            callbacks = kwargs.pop('callbacks', None)
            if callbacks is None:
                callbacks = [early_stop]
            else:
                callbacks.append(early_stop)

            kwargs['callbacks'] = callbacks

        if validation_split > 0:
            split = int(len(label) * validation_split)
            validate_features = {name: arr[-split:]
                                 for name, arr in features.items()}
            validate_label = label[-split:]
            validation_data = (validate_features, validate_label)

            features = {name: arr[:-split] for name, arr in features.items()}
            label = label[:-split]
        else:
            validation_data = None

        self._history = self._model.fit(x=features, y=label, epochs=epochs,
                                        validation_data=validation_data,
                                        **kwargs)

    def predict(self, features, **kwargs):
        """
        Use model to predict label from given features

        Parameters
        ----------
        features : pandas.DataFrame
            features to predict from
        kwargs : dict
            kwargs for tensorflow.keras.model.predict

        Returns
        -------
        pandas.DataFrame
            Label prediction
        """
        features = self._normalize(self._clean_feature_names(features),
                                   means=self._means,
                                   stdevs=self._stdevs)[0]

        features = {name: np.array(value)
                    for name, value in features.items()}

        prediction = pd.DataFrame(self._model.predict(features, **kwargs),
                                  columns=self._label.columns)

        return prediction
