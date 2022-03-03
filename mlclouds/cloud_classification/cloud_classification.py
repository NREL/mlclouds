"""Cloud Classification model using XGBoost"""

import pandas as pd
import xgboost as xgb
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (log_loss,
                             confusion_matrix)
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import tensorflow as tf
tf.random.set_seed(42)

from nsrdb.all_sky.all_sky import all_sky, ALL_SKY_ARGS
from phygnn import TfModel

logger = logging.getLogger(__name__)


class CloudClassificationModel:
    """Cloud Classification Model class using
    XGBoost Classifier to classify conditions into
    clear, water_cloud, and ice_cloud
    """

    # default feature set
    DEF_FEATURES = [
        'solar_zenith_angle',
        'refl_0_65um_nom',
        'refl_0_65um_nom_stddev_3x3',
        'refl_3_75um_nom',
        'temp_3_75um_nom',
        'temp_11_0um_nom',
        'temp_11_0um_nom_stddev_3x3',
        'cloud_probability',
        'cloud_fraction',
        'air_temperature',
        'dew_point',
        'relative_humidity',
        'total_precipitable_water',
        'surface_albedo',
        'alpha',
        'aod',
        'ozone',
        'ssa',
        'surface_pressure',
        'cld_opd_mlclouds_water',
        'cld_opd_mlclouds_ice',
        'cloud_type',
        'flag',
        'cld_opd_dcomp']

    def __init__(self, model_file=None, max_depth=30,
                 n_estimators=2500, test_size=0.2,
                 features=None):
        """Initialize cloud classification model

        Parameters
        ----------
        model_file : str, optional
            file containing previously trained and saved model, by default None
        max_depth : int, optional
            max tree depth for xgboost classifier, by default 30
        n_estimators : int, optional
            number of trees used in xgboost classifier, by default 2500
        test_size : float, optional
            fraction of full data set to reserve for validation, by default 0.2
        features : list, optional
            list of features to use for training and for predictions,
            by default features
        """
        if features is None:
            features = self.DEF_FEATURES

        self.cloud_encoding = {'clearsky': 0, 'water': 1, 'ice': 2}
        self.flag_encoding = {'clear': 0, 'water_cloud': 1,
                              'ice_cloud': 2, 'bad_cloud': 3}

        # UWisc cloud types
        self.cloud_type_encoding = {'clearsky': 0, 'water': 2, 'ice': 6}

        self.initialize_model()

        self.features = features
        self.test_size = test_size
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_indices = None
        self.test_indices = None
        self.model_file = model_file
        self.max_depth = max_depth
        self.n_estimators = n_estimators

    def initialize_model(self):
        """Initialize XGBoost model"""

        if self.model_file is not None:
            try:
                self.model = self.load(self.model_file)
            except FileNotFoundError:
                logger.error(f'Model file not found: {self.model_file}')
        else:
            clf = xgb.XGBClassifier(max_depth=self.max_depth,
                                    n_estimators=self.n_estimators)
            self.model = Pipeline([('scaler', StandardScaler()),
                                   ('clf', clf)])

    def _load_data(self, data_file, frac=None):
        """Load csv data file for training

        Parameters
        ----------
        data_file : str
            csv file containing features and targets for training

        Returns
        -------
        pd.DataFrame
            loaded dataframe with features for training or prediction
        """
        self.df = pd.read_csv(data_file)
        self.df = self.convert_flags(self.df)

        if frac is not None:
            self.df = self.df.groupby(
                'nom_cloud_id').apply(
                    lambda x: x.sample(frac=frac))
        return self.df

    def _select_features(self, df):
        """Extract features from loaded dataframe

        Parameters
        ----------
        df : pd.DataFrame
            dataframe with features to use for cloud type prediction

        Returns
        -------
        X : pd.DataFrame
            dataframe of features to use for training/predictions
        """
        X = df[self.features]
        return X

    def _select_targets(self, df, one_hot_encoding=True):
        """Extract targets from loaded dataframe

        Parameters
        ----------
        df : pd.DataFrame
            dataframe with features to use for cloud type prediction
        one_hot_encoding : bool
            whether to one hot encode targets or keep single column

        Returns
        -------
        y : pd.DataFrame
            dataframe of targets to use for training
        one_hot_coding : bool
            Whether to one hot encode targets or to just
            integer encode
        """
        if one_hot_encoding:
            y = pd.get_dummies(df['nom_cloud_id'])
            self.cloud_encoding = {k: v for v, k in enumerate(y.columns)}
        else:
            y = df['nom_cloud_id'].replace(self.cloud_encoding)
        return y

    def _split_data(self, df, one_hot_encoding=False):
        """Split data into training and validation

        Parameters
        ----------
        df : pd.DataFrame
            dataframe with features to use for cloud type prediction
        one_hot_coding : bool
            Whether to one hot encode targets or to just
            integer encode

        Returns
        -------
        X_train : pd.DataFrame
            Fraction of full feature dataframe to use for training
        X_test : pd.DataFrame
            Fraction of full feature dataframe to use for validation
        y_train : pd.DataFrame
            Fraction of full target dataframe to use for training
        y_test : pd.DataFrame
            Fraction of full target dataframe to use for validation
        """
        X = self._select_features(df)
        y = self._select_targets(df, one_hot_encoding)
        return train_test_split(X, y, np.arange(X.shape[0]),
                                test_size=self.test_size)

    def convert_flags(self, X):
        """Convert the flag column from str to float in an input dataframe
        before prediction

        Parameters
        ----------
        X : pd.DataFrame
            dataframe of features to use for cloud
            type prediction

        Returns
        -------
        X : pd.DataFrame
            Same as input but if 'flag' was found its converted into a float
            column.
        """

        if 'flag' in X:
            try:
                X['flag'] = X['flag'].astype(np.float32)
            except:
                bad_flags = [f for f in X['flag'].unique()
                             if f not in self.flag_encoding.keys()]
                if any(bad_flags):
                    msg = ('The following "flag" values were not found in the '
                           'flag encoding options: {} (available flag '
                           'values: {}'
                           .format(bad_flags, list(self.flag_encoding.keys())))
                    logger.error(msg)
                    raise KeyError(msg)

                X['flag'].replace(self.flag_encoding, inplace=True)
                X['flag'] = X['flag'].astype(np.float32)

        return X

    def check_features(self, X):
        """Check an input dataframe for the features that this model uses.

        Parameters
        ----------
        X : pd.DataFrame
            dataframe of features to use for cloud
            type prediction
        """
        missing = [f for f in self.features if f not in X.columns]
        if any(missing):
            msg = ('The following features were missing from the input '
                   'dataframe: {} '.format(missing))
            logger.error(msg)
            raise ValueError(msg)

    def load_data_and_train(self, data_file, frac=None):
        """Load data and train model using features selected
        during initialization

        Parameters
        ----------
        data_file : str
            csv file containing features and targets for training
        """
        df = self._load_data(data_file=data_file, frac=frac)
        self.X_train, self.X_test, self.y_train, self.y_test, \
            self.train_indices, self.test_indices = self._split_data(df)
        self.train(self.X_train, self.y_train)

    def train(self, X_train, y_train, epochs=None):
        """Train model using provided features and targets

        Parameters
        ----------
        X_train : pd.DataFrame
            Training split of full feature dataframe
        y_train : pd.DataFrame
            Training split of full target dataframe
        """
        if epochs is not None:
            self.model.fit(X_train, y_train, clf__epochs=epochs)
        else:
            self.model.fit(X_train, y_train)

    def save(self, model_file):
        """Save training model

        Parameters
        ----------
        model_file : str
            Output file for saved model
        """
        joblib.dump(self.model, model_file)

    def load(self, model_file):
        """Load model from file

        Parameters
        ----------
        model_file : str
            File where saved model is stored

        Returns
        -------
        model : sklearn.pipeline.Pipeline
            full model pipeline, including scaler
            and classifier
        """
        self.model = joblib.load(model_file)
        return self.model

    def raw_prediction(self, X):
        """Predict cloud type

        Parameters
        ----------
        X : pd.DataFrame
            dataframe of features to use for cloud
            type prediction
        to_cloud_type : bool
            Flag to convert to UWisc numeric cloud types

        Returns
        -------
        y : np.ndarray
            array of cloud type predictions for each
            observation in X
        """

        self.check_features(X)
        X = self.convert_flags(X)
        y = self.model.predict(X[self.features])
        return y

    def remap_predictions(self, X, y, to_cloud_type=False):
        """Remap predictions to string cloud labels

        Parameters
        ----------
        X : pd.DataFrame
            dataframe of features to use for cloud
            type prediction
        y : np.ndarray
            array of cloud type predictions for each
            observation in X
        to_cloud_type : bool
            Flag to convert to UWisc numeric cloud types

        Returns
        -------
        y : np.ndarray
            array of cloud type predictions for each
            observation in X with string labels
        """
        inverse_cloud_encoding = {v: k for k, v in self.cloud_encoding.items()}
        y = [inverse_cloud_encoding[v] for v in y]
        inverse_flag_encoding = {v: k for k, v in self.flag_encoding.items()}
        X['flag'].replace(inverse_flag_encoding, inplace=True)
        if to_cloud_type:
            y = pd.Series(y).map(self.cloud_type_encoding)
            y = y.astype(np.uint16).values
        return y

    def predict(self, X, to_cloud_type=False):
        """Predict cloud type

        Parameters
        ----------
        X : pd.DataFrame
            dataframe of features to use for cloud
            type prediction
        to_cloud_type : bool
            Flag to convert to UWisc numeric cloud types

        Returns
        -------
        y : np.ndarray
            array of cloud type predictions for each
            observation in X
        """

        y = self.raw_prediction(X)
        return self.remap_predictions(X, y, to_cloud_type)

    def update_all_sky_input(self, all_sky_input, df):
        """Update fields for all_sky based on model classifications

        Parameters
        ----------
        all_sky_input : pd.DataFrame
            dataframe to update and then send to all_sky run
        df : pd.DataFrame
            dataframe with variables needed for running all_sky

        Returns
        -------
        pd.DataFrame
            updated all_sky_input with cloud type predictions from
            model classifications
        """

        y = self.predict(df)

        all_sky_input['cloud_type'] = 0
        all_sky_input['cld_opd_dcomp'] = 0
        all_sky_input['cld_reff_dcomp'] = 0

        ice_mask = y == 'ice'
        water_mask = y == 'water'

        all_sky_input.loc[ice_mask, 'cld_opd_dcomp'] = \
            df.loc[ice_mask, 'cld_opd_mlclouds_ice']
        all_sky_input.loc[ice_mask, 'cld_reff_dcomp'] = \
            df.loc[ice_mask, 'cld_reff_mlclouds_ice']
        all_sky_input.loc[ice_mask, 'cloud_type'] = 6
        all_sky_input.loc[water_mask, 'cld_opd_dcomp'] = \
            df.loc[water_mask, 'cld_opd_mlclouds_water']
        all_sky_input.loc[water_mask, 'cld_reff_dcomp'] = \
            df.loc[water_mask, 'cld_reff_mlclouds_water']
        all_sky_input.loc[water_mask, 'cloud_type'] = 2
        all_sky_input['time_index'] = df['time_index'].values
        return all_sky_input

    def run_all_sky(self, df):
        """Update all sky inputs with model predictions and
        then run all_sky

        Parameters
        ----------
        df : pd.DataFrame
            dataframe with features needed to make cloud type
            predictions

        Returns
        -------
        pd.DataFrame
            updated dataframe with all_sky irradiance outputs
        """
        ignore = ('cloud_fill_flag',)

        all_sky_args = [dset for dset in ALL_SKY_ARGS if dset not in ignore]
        all_sky_input = {dset: df[dset].values for dset in all_sky_args}

        all_sky_input = self.update_all_sky_input(all_sky_input, df)

        all_sky_input = {k: np.expand_dims(v, axis=1)
                         for k, v in all_sky_input.items()}

        out = all_sky(**all_sky_input)

        for dset in ('ghi', 'dni', 'dhi'):
            df[f'nn_{dset}'] = out[dset].flatten()

        return df

    def loss(self, X, y):
        """Calculate categorical crossentropy loss
        for given features dataframe and targets

        Parameters
        ----------
        X : pd.DataFrame
            dataframe of features to use for cloud type
            prediction
        y : pd.DataFrame
            dataframe of targets to compare to predictions
            for loss calculation

        Returns
        -------
        loss : float
            categorical crossentropy loss for the given
            predictions and targets
        """
        return log_loss(y, self.model.predict_proba(X))

    def get_confusion_matrix(self, X, y_true, binary=True):
        """Compute confusion matrix from true labels
        and predicted labels

        Parameters
        ----------
        X : pd.DataFrame
            dataframe of features to use for cloud type
            predictions
        y_true : ndarray
            array of cloud type labels
        binary : bool, optional
            whether to compute binary confusion matrix
            (clear/cloudy) or keep all cloud types, by default True

        Returns
        -------
        ndarray
            normalized confusion matrix array
        """
        y_pred = self.predict(X)
        y_pred = np.array([self.cloud_encoding[y] for y in y_pred])

        if len(y_true.shape) > 1:
            y_true = y_true.idxmax(axis=1)
        y_true = np.array(y_true.replace(self.cloud_encoding))
        if binary:
            y_pred[y_pred != 0] = 1
            y_true[y_true != 0] = 1
        cm = confusion_matrix(y_true, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm


class CloudClassificationNN(CloudClassificationModel):
    """Cloud Classification Model class using
    TensorFlow Classifier to classify conditions into
    clear, water_cloud, and ice_cloud
    """
    def __init__(self, model_file=None, test_size=0.2,
                 features=None, learning_rate=0.01,
                 epochs=100, batch_size=128,
                 optimizer='adam'):
        """Initialize cloud classification model

        Parameters
        ----------
        model_file : str, optional
            file containing previously trained and saved model, by default None
        test_size : float, optional
            fraction of full data set to reserve for validation, by default 0.2
        features : list, optional
            list of features to use for training and for predictions,
            by default features
        learning_rate : float, optional
            learning rate for classifier, by default 0.01
        epochs : int
            number of epochs for classifier training
        """
        super().__init__()

        if features is None:
            features = self.DEF_FEATURES

        self.cloud_encoding = {'clearsky': 0, 'water': 1, 'ice': 2}
        self.flag_encoding = {'clear': 0, 'water_cloud': 1,
                              'ice_cloud': 2, 'bad_cloud': 3}
        self.learning_rate = learning_rate
        self.epochs = epochs

        # UWisc cloud types
        self.cloud_type_encoding = {'clearsky': 0, 'water': 2, 'ice': 6}

        self.initialize_model()

        self.optimizer = optimizer
        self.model_file = model_file
        self.features = features
        self.test_size = test_size
        self.batch_size = batch_size
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_indices = None
        self.test_indices = None

    def initialize_model(self):
        """Initialize model architecture"""

        if self.model_file is not None:
            try:
                self.model = self.load(self.model_file)
            except FileNotFoundError:
                logger.error(f'Model file not found: {self.model_file}')
        else:
            clf = tf.keras.models.Sequential()
            clf.add(tf.keras.layers.Normalization())
            clf.add(tf.keras.layers.Dense(128, activation='relu'))
            clf.add(tf.keras.layers.Dense(128, activation='relu'))
            clf.add(tf.keras.layers.Dense(128, activation='relu'))
            clf.add(tf.keras.layers.Dense(128, activation='relu'))
            clf.add(tf.keras.layers.Dense(128, activation='relu'))
            clf.add(tf.keras.layers.Dense(3, activation='sigmoid'))

            if self.optimizer == 'adam':
                opt = tf.keras.optimizers.Adam(
                    learning_rate=self.learning_rate
                )
            if self.optimizer == 'sgd':
                opt = tf.keras.optimizers.SGD(
                    learning_rate=self.learning_rate
                )
            clf.compile(
                loss=tf.keras.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=[
                    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                ]
            )
            self.model = clf

    def train(self, X_train, y_train, X_test, y_test):
        """
        Parameters
        ----------
        X : pd.DataFrame
            dataframe of features to use for cloud
            type prediction
        y : pd.DataFrame
            dataframe of targets to use for cloud
            type prediction

        Returns
        -------
        history : dict
            dictionary with loss and accuracy history
            over course of training
        """
        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=5,
            restore_best_weights=True)

        history = self.model.fit(
            X_train, y_train, batch_size=self.batch_size,
            epochs=self.epochs, validation_data=(X_test, y_test),
            callbacks=[earlystopping])
        return history['clf'].history.history

    def load_data_and_train(self, data_file, frac=None):
        """Load data and train model using features selected
        during initialization

        Parameters
        ----------
        data_file : str
            csv file containing features and targets for training

        Returns
        -------
        history : dict
            dictionary with loss and accuracy history
            over course of training
        """
        df = self._load_data(data_file=data_file, frac=frac)
        self.X_train, self.X_test, self.y_train, self.y_test, \
            self.train_indices, self.test_indices = self._split_data(
                df, one_hot_encoding=True)
        return self.train(
            self.X_train, self.y_train, self.X_test, self.y_test)

    def predict(self, X, to_cloud_type=False):
        """Predict cloud type

        Parameters
        ----------
        X : pd.DataFrame
            dataframe of features to use for cloud
            type prediction
        to_cloud_type : bool
            Flag to convert to UWisc numeric cloud types

        Returns
        -------
        y : np.ndarray
            array of cloud type predictions for each
            observation in X
        """

        y = self.raw_prediction(X)
        y = pd.DataFrame(y)
        y = y.idxmax(axis=1)

        return self.remap_predictions(X, y, to_cloud_type)
