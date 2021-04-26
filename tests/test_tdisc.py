"""
Test the tensor implementation of disc
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from farms.disc import disc
from mlclouds.tdisc import tdisc


def test_tensor_disc():
    """Run normal disc and tensor disc and compare the outputs."""

    ghi = np.array([[5, 200, 100, 500, 1100]], dtype=np.float32).T
    tghi = tf.convert_to_tensor(ghi, dtype=tf.float32)

    time_index = pd.date_range('20180101', '20190101', freq='1h')[0:5]
    doy = time_index.dayofyear.values
    sza = np.array([[45, 30, 10, 0, 10]]).T

    baseline_dni = disc(ghi, sza, doy, pressure=101325)
    tensor_dni = tdisc(tghi, sza, doy, pressure=101325)

    assert tf.is_tensor(tensor_dni)
    assert np.allclose(baseline_dni, tensor_dni.numpy())
