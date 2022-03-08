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
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy

tf.random.set_seed(42)

from phygnn import TfModel

from nsrdb.all_sky.all_sky import ALL_SKY_ARGS, all_sky


logger = logging.getLogger(__name__)


def get_confusion_matrix(y_pred, y_true, binary=True):
    """Compute confusion matrix from true labels
    and predicted labels

    Parameters
    ----------
    y_pred : ndarray
        array of predicted cloud type labels
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
    if binary:
        y_pred[y_pred != 0] = 1
        y_true[y_true != 0] = 1
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


def normalize(X, means=None, stds=None):
    """Normalize features used computed stats
    or inputs

    Parameters
    ----------
    X : pd.DataFrame
        dataframe of features
    means : ndarray, optional
        array of means for each feature, by default None
    stds : ndarray, optional
        array of standard deviations for each
        feature, by default None

    Returns
    -------
    _type_
        _description_
    """
    if means is None:
        means = X.mean(axis=0)

    if stds is None:
        stds = X.std(axis=0)

    return (X - means) / stds


def plot_binary_cm(cm, title='Confusion Matrix'):
    """Plot confusion matrix for cloudy/clear

    Parameters
    ----------
    cm : ndarray
        binary confusion matrix
    title : str, optional
        Title of confusion matrix plot, by default 'Confusion Matrix'
    """
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(title)
    ax.yaxis.set_ticklabels(['clear', 'cloudy'])
    ax.xaxis.set_ticklabels(['clear', 'cloudy'])
    plt.show()


def update_dataframe(df, y):
    """Update fields for all_sky based on model classifications

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of features to use for cloud type
        predictions
    y : ndarray
        array of cloud type predictions

    Returns
    -------
    pd.DataFrame
        updated dataframe with cloud type predictions from
        model classifications
    """

    df = df.reset_index(drop=True)

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

    df['cloud_id_model'] = y

    return df


def remap_predictions(y, cloud_type_encoding=None):
    """Remap predictions to string cloud labels

    Parameters
    ----------
    y : pd.DataFrame
        dataframe of cloud type predictions

    Returns
    -------
    y_pred : np.ndarray
        array of cloud type predictions
    """

    if cloud_type_encoding is not None:
        y_pred = y.replace({v: k for k, v in cloud_type_encoding.items()})
    else:
        y_pred = y.idxmax(axis=1)
    return y_pred


def encode_features(X, features):
    """One hot encode features

    Parameters
    ----------
    X : pd.DataFrame
        dataframe of features
    features : list
        list of features to extract from X

    Returns
    -------
    pd.DataFrame
        dataframe of encoded features
    """
    X_new = pd.get_dummies(X[features])
    return X_new


def encode_predictions(y, cloud_type_encoding=None):
    """Remap predictions to integer cloud labels

    Parameters
    ----------
    y : pd.DataFrame
        dataframe of cloud type predictions

    Returns
    -------
    y_pred : np.ndarray
        array of cloud type predictions
    """

    if cloud_type_encoding is not None:
        y_pred = y.replace(cloud_type_encoding)
    else:
        y_pred = y.idxmax(axis=1)
        y_pred = y_pred.replace({k: v for v, k in enumerate(y.columns)})
    return y_pred


def run_all_sky(df, y):
    """Update all sky inputs with model predictions and
    then run all_sky

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of features to use for cloud type
        predictions
    y : ndarray
        array of cloud type predictions

    Returns
    -------
    pd.DataFrame
        updated dataframe with all_sky irradiance outputs
    """
    ignore = ('cloud_fill_flag',)

    all_sky_args = [dset for dset in ALL_SKY_ARGS if dset not in ignore]
    all_sky_input = {dset: df[dset].values for dset in all_sky_args}

    df = update_dataframe(df, y)

    all_sky_input['cloud_type'] = df['cloud_type'].values
    all_sky_input['cld_opd_dcomp'] = df['cld_opd_dcomp'].values
    all_sky_input['cld_reff_dcomp'] = df['cld_reff_dcomp'].values
    all_sky_input = {k: np.expand_dims(v, axis=1)
                     for k, v in all_sky_input.items()}
    all_sky_input['time_index'] = df['time_index'].values

    out = all_sky(**all_sky_input)

    for dset in ('ghi', 'dni', 'dhi'):
        df[f'{dset}_model'] = out[dset].flatten()

    return df


class CloudClassificationBase:
    """Base Class for cloud classifier"""

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

    DEF_LABELS = [
        'clearsky',
        'ice',
        'water'
    ]

    def __init__(self, **kwargs):
        self.model = self.initialize_model(**kwargs)

    @staticmethod
    def initialize_model(learning_rate=0.001):
        """Initialize sequential model layers and compile

        Parameters
        ----------
        learning_rate : float
            model learning rate

        Returns
        -------
        Sequential
            sequential tensorflow model
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(3, activation='sigmoid'))

        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')])

        return model

    @staticmethod
    def load_data(data_file, frac=None):
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
        df = pd.read_csv(data_file)

        if frac is not None:
            df = df.groupby(
                'nom_cloud_id', group_keys=False).apply(
                    lambda x: x.sample(frac=frac))
        return df

    @staticmethod
    def select_features(df, features):
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
        X = encode_features(df, features)
        return normalize(X)

    @staticmethod
    def select_targets(df):
        """Extract targets from loaded dataframe

        Parameters
        ----------
        df : pd.DataFrame
            dataframe with features to use for cloud type prediction

        Returns
        -------
        y : pd.DataFrame
            dataframe of targets to use for training
        """
        y = pd.get_dummies(df['nom_cloud_id'])
        return y[CloudClassificationBase.DEF_LABELS]

    def split_data(self, df, features):
        """Split data into training and validation

        Parameters
        ----------
        df : pd.DataFrame
            dataframe of features and targets

        Returns
        -------
        X_train : pd.DataFrame
            dataframe of features to use for training
        X_test : pd.DataFrame
            dataframe of feeatures to use for validation
        y_train : pd.DataFrame
            dataframe of targets to use for training
        y_test : pd.DataFrame
            dataframe of targets to use for validation
        """
        X = self.select_features(df, features)
        y = self.select_targets(df)
        return train_test_split(
            X, y, test_size=0.2, random_state=42)

    def load_and_train(self, data_file):
        """Load data and train model

        Parameters
        ----------
        data_file : str
            path to data file with feaures and targets

        Returns
        -------
        dict
            dictionary with training history
        """

        df = self.load_data(data_file)
        X_train, X_test, y_train, y_test = self.split_data(
            df, self.DEF_FEATURES)
        history = self.train_model(X_train, X_test, y_train, y_test)
        return history

    def train_model(self, X_train, X_test, y_train, y_test):
        """Train model using provided training and test data

        Parameters
        ----------
        X_train : pd.DataFrame
            dataframe of features to use for training
        X_test : pd.DataFrame
            dataframe of feeatures to use for validation
        y_train : pd.DataFrame
            dataframe of targets to use for training
        y_test : pd.DataFrame
            dataframe of targets to use for validation

        Returns
        -------
        history : dict
            history of model training
        """

        history = self.model.fit(
            X_train, y_train, epochs=100,
            validation_data=(X_test, y_test),
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0, patience=10, verbose=0,
                mode='min', baseline=None, restore_best_weights=True)])

        return history

    def tune_learning_rate(self, X, y):
        """Optimize learning rate

        Parameters
        ----------
        X : pd.DataFrame
            dataframe of features
        y : pd.DataFrame
            dataframe of targets

        Returns
        -------
        history : dict
            history of model training
        """
        initial_history = self.model.fit(
            X, y, epochs=100, callbacks=[
                tf.keras.callbacks.LearningRateScheduler(
                    lambda epoch: 1e-4 * 10 ** (epoch / 30))])
        return initial_history

    def predict(self, df):
        """Predict new cloud type labels

        Parameters
        ----------
        df : pd.DataFrame
            dataframe of features to use for predictions


        Returns
        -------
        y : ndarray
            array of cloud type labels
        """
        X_scaled = self.select_features(df, self.DEF_FEATURES)
        y_pred = self.model.predict(X_scaled)
        y_pred = y_pred.idxmax(axis=1)
        y_pred = remap_predictions(
            y_pred, {k: v for k, v in enumerate(self.DEF_LABELS)})
        return y_pred

    def load_train_and_run_all_sky(self, data_file):
        """Load and train model then run all sky with
        predictions

        Parameters
        ----------
        data_file : str
            path to data file with features and targets for training

        Returns
        -------
        pd.DataFrame
            dataframe with cloud type predictions and irradiance from all sky
        """
        df = self.load_data(data_file)
        X_train, X_test, y_train, y_test = self.split_data(df)
        _ = self.train_model(X_train, X_test, y_train, y_test)
        y_pred = self.predict(df)
        df_res = run_all_sky(df, y_pred)
        return df_res


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
        self.model = None

        self.initialize_model()

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

    def load_data(self, data_file, frac=None):
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

    def select_features(self, df, features=DEF_FEATURES):
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
        X = df[features]
        return X

    def select_targets(self, df):
        """Extract targets from loaded dataframe

        Parameters
        ----------
        df : pd.DataFrame
            dataframe with features to use for cloud type prediction

        Returns
        -------
        y : pd.DataFrame
            dataframe of targets to use for training
        one_hot_coding : bool
            Whether to one hot encode targets or to just
            integer encode
        """
        y = df['nom_cloud_id'].replace(self.cloud_encoding)
        return y

    def split_data(self, df, features=DEF_FEATURES):
        """Split data into training and validation

        Parameters
        ----------
        df : pd.DataFrame
            dataframe with features to use for cloud type prediction

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
        train_indices : ndarray
            array of indices corresponding to training data
        test_indices : ndarray
            array of indices corresponding to test data
        """
        X = self.select_features(df, features)
        y = self.select_targets(df)
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

    def load_data_and_split(self, data_file, frac=None):
        """Load data and train model using features selected
        during initialization

        Parameters
        ----------
        data_file : str
            csv file containing features and targets for training

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
        train_indices : ndarray
            array of indices corresponding to training data
        test_indices : ndarray
            array of indices corresponding to test data
        """
        df = self.load_data(data_file=data_file, frac=frac)
        return self.split_data(df)

    def load_data_and_train(self, data_file, frac=None):
        """Load data and train model using features selected
        during initialization

        Parameters
        ----------
        data_file : str
            csv file containing features and targets for training
        """
        df = self.load_data(data_file=data_file, frac=frac)
        self.X_train, self.X_test, self.y_train, self.y_test, \
            self.train_indices, self.test_indices = self.split_data(df)
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


class CloudClassificationNN(TfModel):
    """Cloud Classification Model class using
    TensorFlow Classifier to classify conditions into
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

    def __init__(self, model, feature_names,
                 label_names, **kwargs):
        """Initialize cloud classification model

        Parameters
        ----------
        model_file : str, optional
            file containing previously trained and saved model, by default None
        data_file : str
            file containing features and targets to use for training
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

        self.X = None
        self.y = None
        self.df = None
        super().__init__(model, feature_names,
                         label_names, **kwargs)

    @staticmethod
    def load_data(data_file, frac=None):
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
        df = pd.read_csv(data_file)

        if frac is not None:
            df = df.groupby(
                'nom_cloud_id', group_keys=False).apply(
                    lambda x: x.sample(frac=frac))
        return df

    @staticmethod
    def select_features(df, features):
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
        X = pd.get_dummies(df[features])
        return X

    @staticmethod
    def select_targets(df):
        """Extract targets from loaded dataframe

        Parameters
        ----------
        df : pd.DataFrame
            dataframe with features to use for cloud type prediction

        Returns
        -------
        y : pd.DataFrame
            dataframe of targets to use for training
        """
        y = pd.get_dummies(df['nom_cloud_id'])
        return y

    @staticmethod
    def initialize_layers():
        """Initialize model architecture"""

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(3, activation='sigmoid'))

        return model

    @classmethod
    def initialize_model(cls, data_file, frac=None):
        """Load data and initialize TfModel

        Parameters
        ----------
        data_file : str
            file with feature and target data
        frac : float, optional
            fraction of full dataset to use, by default None

        Returns
        -------
        TfModel
            TfModel class initialized with feature and
            label names from data file
        """
        df = cls.load_data(data_file=data_file, frac=frac)
        X = cls.select_features(df, cls.DEF_FEATURES)
        y = cls.select_targets(df)
        model = cls.initialize_layers()
        clf = cls(model, X.columns, y.columns)
        clf.df = df
        clf.X = X
        clf.y = y
        return clf

    @classmethod
    def initialize_and_train(cls, data_file, frac=None, **kwargs):
        """Load data and initialize TfModel

        Parameters
        ----------
        data_file : str
            file with feature and target data
        frac : float, optional
            fraction of full dataset to use, by default None

        Returns
        -------
        TfModel
            Initialized and trained TfModel
        """
        clf = cls.initialize_model(
            data_file=data_file, frac=frac)

        if not kwargs:
            kwargs = {}

        kwargs['loss'] = kwargs.get('loss', 'categorical_crossentropy')
        kwargs['metrics'] = kwargs.get(
            'metrics', [categorical_crossentropy, categorical_accuracy])
        model = clf.build_trained(clf.X, clf.y, **kwargs)
        model.df = clf.df
        model.X = clf.X
        model.y = clf.y
        return model

    def predict_new(self, df):
        """Predict labels from new dataframe

        Parameters
        ----------
        df : pd.DataFrame
            dataframe of features to use for predictions

        Returns
        -------
        ndarray
            array of cloud type predictions
        """

        X = self.select_features(df, self.DEF_FEATURES)
        y_pred = self.predict(pd.get_dummies(X)[self.feature_names])
        y_pred = remap_predictions(y_pred)
        return y_pred
