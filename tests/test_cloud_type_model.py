# -*- coding: utf-8 -*-
"""
Test loading and running of the saved production mlclouds model
"""

import numpy as np
import pandas as pd
from farms import CLEAR_TYPES, ICE_TYPES, WATER_TYPES

from mlclouds import CTYPE_MODEL_FPATH
from mlclouds.model.base import MLCloudsModel

# CPROP_MODEL_FPATH = '/kfs2/projects/pxs/mlclouds/nasa_model/cloud_prop_nostd_more/outputs/mlclouds_model.pkl'


def test_clear_cloud_type_predictions():
    """Test that the cloud type model produces reasonable outputs for zero
    reflectance"""

    defaults = {
        'solar_zenith_angle': 10,
        'refl_0_65um_nom': 0,
        'temp_3_75um_nom': 0,
        'temp_11_0um_nom': 0,
        'air_temperature': 10,
        'dew_point': 10,
        'relative_humidity': 80,
        'total_precipitable_water': 5,
        'surface_albedo': 0.1,
        'cld_press_acha': 800,
    }

    model = MLCloudsModel.load(CTYPE_MODEL_FPATH)
    assert any(fn in model.feature_names for fn in defaults)
    ctypes = (
        'clear_fraction',
        'ice_fraction',
        'water_fraction',
    )
    output_names = ('cloud_type',)
    assert all(ct in model.label_names for ct in ctypes)
    assert all(on in model.output_names for on in output_names)

    assert len(model.history) >= 190
    assert model.history['training_loss'].values[-1] < 1

    missing = [fn for fn in model.feature_names if fn not in defaults]
    if any(missing):
        msg = 'Need to update test with default feature inputs for {}.'.format(
            missing
        )
        raise KeyError(msg)

    features = {fn: defaults[fn] * np.ones(10) for fn in model.feature_names}
    features = pd.DataFrame(features)

    out = model.predict(features, table=True)

    assert out['cloud_type'].isin(CLEAR_TYPES).all()


def test_water_cloud_type_predictions():
    """Test that the cloud type model produces reasonable outputs for water
    cloud conditions"""

    defaults = {
        'solar_zenith_angle': 10,
        'refl_0_65um_nom': 90,
        'temp_3_75um_nom': 300,
        'temp_11_0um_nom': 300,
        'air_temperature': 10,
        'dew_point': 10,
        'relative_humidity': 80,
        'total_precipitable_water': 5,
        'surface_albedo': 0.1,
    }

    model = MLCloudsModel.load(CTYPE_MODEL_FPATH)
    assert any(fn in model.feature_names for fn in defaults)
    ctypes = (
        'clear_fraction',
        'ice_fraction',
        'water_fraction',
    )
    output_names = ('cloud_type',)
    assert all(ct in model.label_names for ct in ctypes)
    assert all(on in model.output_names for on in output_names)

    assert len(model.history) >= 190
    assert model.history['training_loss'].values[-1] < 1

    missing = [fn for fn in model.feature_names if fn not in defaults]
    if any(missing):
        msg = 'Need to update test with default feature inputs for {}.'.format(
            missing
        )
        raise KeyError(msg)

    features = {fn: defaults[fn] * np.ones(10) for fn in model.feature_names}
    features = pd.DataFrame(features)

    out = model.predict(features, table=True)

    assert out['cloud_type'].isin(WATER_TYPES).all()


def test_ice_cloud_type_predictions():
    """Test that the cloud type model produces reasonable outputs for ice cloud
    conditions"""

    defaults = {
        'solar_zenith_angle': 10,
        'refl_0_65um_nom': 200,
        'temp_3_75um_nom': 100,
        'temp_11_0um_nom': 100,
        'air_temperature': 2,
        'dew_point': 10,
        'relative_humidity': 10,
        'total_precipitable_water': 10,
        'surface_albedo': 0.1,
    }

    model = MLCloudsModel.load(CTYPE_MODEL_FPATH)
    assert any(fn in model.feature_names for fn in defaults)
    ctypes = (
        'clear_fraction',
        'ice_fraction',
        'water_fraction',
    )
    output_names = ('cloud_type',)
    assert all(ct in model.label_names for ct in ctypes)
    assert all(on in model.output_names for on in output_names)

    assert len(model.history) >= 190
    assert model.history['training_loss'].values[-1] < 1

    missing = [fn for fn in model.feature_names if fn not in defaults]
    if any(missing):
        msg = 'Need to update test with default feature inputs for {}.'.format(
            missing
        )
        raise KeyError(msg)

    features = {fn: defaults[fn] * np.ones(10) for fn in model.feature_names}
    features = pd.DataFrame(features)

    out = model.predict(features, table=True)

    assert out['cloud_type'].isin(ICE_TYPES).all()
