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

logger = logging.getLogger(__name__)


class CloudClassificationModel:
    """Cloud Classification Model class using
    XGBoost Classifier to classify conditions into
    clear, water_cloud, and ice_cloud
    """

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
            self.features = [
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
                'flag']

        self.cloud_encoding = {'clearsky': 0, 'water': 1, 'ice': 2}
        self.flag_encoding = {'clear': 0, 'water_cloud': 1,
                              'ice_cloud': 2, 'bad_cloud': 3}

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

    def _load_data(self, data_file):
        """Load csv data file for training

        Parameters
        ----------
        data_file : str
            csv file containing features and targets for training
        """
        self.df = pd.read_csv(data_file)
        self.df['flag'] = self.df['flag'].replace(self.flag_encoding)

    def _select_features(self):
        """Extract features from loaded dataframe

        Returns
        -------
        X : pd.DataFrame
            dataframe of features to use for training/predictions
        """
        X = self.df[self.features]
        return X

    def _select_targets(self):
        """Extract targets from loaded dataframe

        Returns
        -------
        y : pd.DataFrame
            dataframe of targets to use for training
        """
        self.df['nom_cloud_id'].replace(self.cloud_encoding, inplace=True)
        return self.df['nom_cloud_id']

    def _split_data(self):
        """Split data into training and validation

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
        y = self._select_targets()
        return train_test_split(X, y, np.arange(X.shape[0]),
                                test_size=self.test_size)

    def load_data_and_train(self, data_file):
        """Load data and train model using features selected
        during initialization

        Parameters
        ----------
        data_file : str
            csv file containing features and targets for training
        """
        self._load_data(data_file=data_file)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self._split_data()
        self.train(self.X_train, self.y_train)

    def train(self, X_train, y_train):
        """Train model using provided features and targets

        Parameters
        ----------
        X_train : pd.DataFrame
            Training split of full feature dataframe
        y_train : pd.DataFrame
            Training split of full target dataframe
        """
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

    def predict(self, X):
        """Predict cloud type

        Parameters
        ----------
        X : pd.DataFrame
            dataframe of features to use for cloud
            type prediction

        Returns
        -------
        y : np.ndarray
            array of cloud type predictions for each
            observation in X
        """

        if 'flag' in X:
            X['flag'].replace(self.flag_encoding, inplace=True)
        y = self.model.predict(X[self.features])
        inverse_cloud_encoding = {v: k for k, v in self.cloud_encoding}
        y = [inverse_cloud_encoding[v] for v in y]
        inverse_flag_encoding = {v: k for k, v in self.flag_encoding}
        X['flag'].replace(inverse_flag_encoding, inplace=True)
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
