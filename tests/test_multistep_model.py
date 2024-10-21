# -*- coding: utf-8 -*-
"""
Test loading and running of the saved production mlclouds model
"""

import numpy as np
import pandas as pd

from mlclouds import CPROP_MODEL_FPATH, CTYPE_MODEL_FPATH
from mlclouds.model.multi_step import MultiCloudsModel


def test_multistep_load_and_run():
    """Test that the multistep cloud type prediction followed by cloud property
    prediction model loads and runs a dummy prediction"""

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
        'cld_press_acha': 800,
    }

    model = MultiCloudsModel.load(CPROP_MODEL_FPATH, CTYPE_MODEL_FPATH)
    assert any(fn in model.feature_names for fn in defaults)
    cprops = (
        'cld_opd_dcomp',
        'cld_reff_dcomp',
    )
    ctypes = (
        'clear_fraction',
        'ice_fraction',
        'water_fraction',
    )
    output_names = ('cld_opd_dcomp', 'cld_reff_dcomp', 'cloud_type')
    assert all(cp in model.label_names for cp in cprops)
    assert all(ct in model.cloud_type_model.label_names for ct in ctypes)
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

    assert (out['cld_opd_dcomp'] > 0).all()
    assert (out['cld_opd_dcomp'] < 100).all()

    assert (out['cld_reff_dcomp'] > 0).all()
    assert (out['cld_reff_dcomp'] < 20).all()

    assert out['cloud_type'].isin([0, 3, 6]).all()


def test_just_prop_load_and_run():
    """Test that the multistep model with only cloud property model loads and
    runs a dummy prediction"""

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
        'clear': 0,
        'ice_cloud': 0,
        'water_cloud': 1,
        'bad_cloud': 0,
    }

    model = MultiCloudsModel.load(CPROP_MODEL_FPATH)

    assert any(fn in model.feature_names for fn in defaults)
    assert 'cld_opd_dcomp' in model.label_names
    assert 'cld_reff_dcomp' in model.label_names

    assert 'cld_opd_dcomp' in model.output_names
    assert 'cld_reff_dcomp' in model.output_names

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

    assert (out['cld_opd_dcomp'] > 0).all()
    assert (out['cld_opd_dcomp'] < 100).all()

    assert (out['cld_reff_dcomp'] > 0).all()
    assert (out['cld_reff_dcomp'] < 20).all()
