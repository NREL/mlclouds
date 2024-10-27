# -*- coding: utf-8 -*-
"""
Test loading and running of the saved production mlclouds model
"""

import numpy as np
import pandas as pd
from farms import CLOUD_TYPES

from mlclouds import COMBINED_MODEL_FPATH
from mlclouds.model.base import MLCloudsModel


def test_load_and_run():
    """Test that the combined cloud type plus cloud properties mlclouds model
    loads and runs a dummy prediction"""

    defaults = {
        'solar_zenith_angle': 10,
        'refl_0_65um_nom': 90,
        'refl_3_75um_nom': 90,
        'temp_3_75um_nom': 300,
        'temp_11_0um_nom': 300,
        'air_temperature': 10,
        'dew_point': 10,
        'cld_press_acha': 800,
        'relative_humidity': 80,
        'total_precipitable_water': 5,
        'surface_albedo': 0.1,
    }

    model = MLCloudsModel.load(COMBINED_MODEL_FPATH)

    assert any(fn in model.feature_names for fn in defaults)
    label_names = (
        'cld_opd_dcomp',
        'cld_reff_dcomp',
        'clear_fraction',
        'ice_fraction',
        'water_fraction',
    )
    output_names = ('cld_opd_dcomp', 'cld_reff_dcomp', 'cloud_type')
    assert all(ln in model.label_names for ln in label_names)
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

    assert out['cloud_type'].isin(CLOUD_TYPES).all()
