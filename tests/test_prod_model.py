# -*- coding: utf-8 -*-
"""
Test loading and running of the saved production mlclouds model
"""
import numpy as np
import pandas as pd

from mlclouds import MODEL_FPATH
from phygnn import PhygnnModel


def test_load_and_run():
    """Test that the mlclouds model loads and runs a dummy prediction"""

    defaults = {'solar_zenith_angle': 10,
                'refl_0_65um_nom': 90,
                'refl_0_65um_nom_stddev_3x3': 1,
                'refl_3_75um_nom': 90,
                'temp_3_75um_nom': 300,
                'temp_11_0um_nom': 300,
                'temp_11_0um_nom_stddev_3x3': 10,
                'cloud_probability': 0.9,
                'cloud_fraction': 0.9,
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

    model = PhygnnModel.load(MODEL_FPATH)

    assert any(fn in model.feature_names for fn in defaults.keys())
    assert 'cld_opd_dcomp' in model.label_names
    assert 'cld_reff_dcomp' in model.label_names
    assert len(model.history) >= 190
    assert model.history['training_loss'].values[-1] < 1

    missing = [fn for fn in model.feature_names if fn not in defaults]
    if any(missing):
        msg = ('Need to update test with default feature inputs for {}.'
               .format(missing))
        raise KeyError(msg)

    features = {fn: defaults[fn] * np.ones(10) for fn in model.feature_names}
    features = pd.DataFrame(features)

    out = model.predict(features, table=True)

    assert (out['cld_opd_dcomp'] > 0).all()
    assert (out['cld_opd_dcomp'] < 10).all()

    assert (out['cld_reff_dcomp'] > 0).all()
    assert (out['cld_reff_dcomp'] < 10).all()
