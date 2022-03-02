"""Cloud Classification model using XGBoost"""

import pandas as pd
import xgboost as xgb
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import tensorflow as tf

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

        if model_file is not None:
            try:
                self.model = self.load(model_file)
            except FileNotFoundError:
                logger.error(f'Model file not found: {model_file}')
        else:
            clf = xgb.XGBClassifier(max_depth=max_depth,
                                    n_estimators=n_estimators)
            self.model = Pipeline([('scaler', StandardScaler()),
                                   ('clf', clf)])
        self.features = features
        self.test_size = test_size
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_indices = None
        self.test_indices = None

    def _load_data(self, data_file):
        """Load csv data file for training

        Parameters
        ----------
        data_file : str
            csv file containing features and targets for training
        """
        self.df = pd.read_csv(data_file)
        self.df = self.convert_flags(self.df)

    def _select_features(self):
        """Extract features from loaded dataframe

        Returns
        -------
        X : pd.DataFrame
            dataframe of features to use for training/predictions
        """
        X = self.df[self.features]
        return X

    def _select_targets(self, one_hot_encoding=True):
        """Extract targets from loaded dataframe

        Returns
        -------
        y : pd.DataFrame
            dataframe of targets to use for training
        one_hot_coding : bool
            Whether to one hot encode targets or to just
            integer encode
        """
        if one_hot_encoding:
            y = pd.get_dummies(self.df['nom_cloud_id'])
            self.cloud_encoding = {k: v for v, k in enumerate(y.columns)}
        else:
            y = self.df['nom_cloud_id'].replace(self.cloud_encoding)
        return y

    def _split_data(self, one_hot_encoding=False):
        """Split data into training and validation

        Parameters
        ----------
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
        X = self._select_features()
        y = self._select_targets(one_hot_encoding)
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

    def load_data_and_train(self, data_file):
        """Load data and train model using features selected
        during initialization

        Parameters
        ----------
        data_file : str
            csv file containing features and targets for training
        """
        self._load_data(data_file=data_file)
        self.X_train, self.X_test, self.y_train, self.y_test, \
            self.train_indices, self.test_indices = self._split_data()
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

        self.check_features(X)
        X = self.convert_flags(X)
        y = self.model.predict(X[self.features])

        inverse_cloud_encoding = {v: k for k, v in self.cloud_encoding.items()}
        y = [inverse_cloud_encoding[v] for v in y]
        inverse_flag_encoding = {v: k for k, v in self.flag_encoding.items()}
        X['flag'].replace(inverse_flag_encoding, inplace=True)
        if to_cloud_type:
            y = pd.Series(y).map(self.cloud_type_encoding)
            y = y.astype(np.uint16).values
        return y

    def update_all_sky_input(self, all_sky_input):
        """Update fields for all_sky based on model classifications

        Parameters
        ----------
        all_sky_input : pd.DataFrame
            dataframe with variables needed for running all_sky

        Returns
        -------
        pd.DataFrame
            updated all_sky_input with cloud type predictions from
            model classifications
        """

        df = all_sky_input.copy()
        y = self.predict(all_sky_input)

        df['cloud_type'] = 0
        df['cld_opd_dcomp'] = 0
        df['cld_reff_dcomp'] = 0

        ice_mask = y == 'ice'
        water_mask = y == 'water'

        df.loc[ice_mask, 'cld_opd_dcomp'] = \
            df.loc[ice_mask, 'cld_opd_mlclouds_ice']
        df.loc[ice_mask, 'cld_reff_dcomp'] = \
            df.loc[ice_mask, 'cld_reff_mlclouds_ice']
        df.loc[ice_mask, 'cloud_type'] = 6
        df.loc[water_mask, 'cld_opd_dcomp'] = \
            df.loc[water_mask, 'cld_opd_mlclouds_water']
        df.loc[water_mask, 'cld_reff_dcomp'] = \
            df.loc[water_mask, 'cld_reff_mlclouds_water']
        df.loc[water_mask, 'cloud_type'] = 2
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


class CloudClassificationNN(CloudClassificationModel):
    """Cloud Classification Model class using
    TensorFlow Classifier to classify conditions into
    clear, water_cloud, and ice_cloud
    """
    def __init__(self, model_file=None, test_size=0.2,
                 features=None, learning_rate=0.01,
                 epochs=100, batch_size=128):
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

        if model_file is not None:
            try:
                self.model = self.load(model_file)
            except FileNotFoundError:
                logger.error(f'Model file not found: {model_file}')
        else:
            clf = tf.keras.models.Sequential()
            clf.add(tf.keras.layers.Dense(128, activation='relu'))
            clf.add(tf.keras.layers.Dense(256, activation='relu'))
            clf.add(tf.keras.layers.Dense(256, activation='relu'))
            clf.add(tf.keras.layers.Dense(3, activation='sigmoid'))
            clf.summary()
            opt = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate)
            clf.compile(
                loss=tf.keras.losses.binary_crossentropy,
                optimizer=opt,
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                ]
            )
            self.model = Pipeline([('scaler', StandardScaler()),
                                   ('clf', clf)])
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
        return history

    def load_data_and_train(self, data_file):
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
        self._load_data(data_file=data_file)
        self.X_train, self.X_test, self.y_train, self.y_test, \
            self.train_indices, self.test_indices = self._split_data(
                one_hot_encoding=True)
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

        self.check_features(X)
        X = self.convert_flags(X)
        y = self.model.predict(X[self.features])
        y = pd.DataFrame(y)
        y = y.idxmax(axis=1)

        inverse_cloud_encoding = {v: k for k, v in self.cloud_encoding.items()}
        y = [inverse_cloud_encoding[v] for v in y]
        inverse_flag_encoding = {v: k for k, v in self.flag_encoding.items()}
        X['flag'].replace(inverse_flag_encoding, inplace=True)
        if to_cloud_type:
            y = pd.Series(y).map(self.cloud_type_encoding)
            y = y.astype(np.uint16).values
        return y
