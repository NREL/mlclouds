"""
Test the tensor implementation of FARMS.
"""
import tensorflow as tf
import numpy as np
from farms.farms import farms
from mlclouds.tfarms import tfarms


def test_tensor_farms():
    """Run normal farms and tensor farms and compare the outputs."""

    tau = np.array([[0.5, 5, 10, 20, 4.4, 3.2, 0.5]]).T
    ttau = tf.convert_to_tensor(tau, dtype=tf.float32)

    cloud_type = np.array([[4, 2, 3, 8, 9, 9, 8]]).T
    cloud_radius = np.array([[2, 3, 4, 10, 23, 32, 5]]).T
    n = len(tau)
    sza = np.ones((n, 1)) * 20
    radius = np.ones((n, 1))
    Tuuclr = np.ones((n, 1)) * 0.5
    Ruuclr = np.ones((n, 1)) * 0.5
    Tddclr = np.ones((n, 1)) * 0.5
    Tduclr = np.ones((n, 1)) * 0.5
    albedo = np.ones((n, 1)) * 0.5

    baseline_ghi = farms(tau, cloud_type, cloud_radius, sza, radius,
                         Tuuclr, Ruuclr, Tddclr, Tduclr, albedo, debug=False)
    tensor_ghi = tfarms(ttau, cloud_type, cloud_radius, sza, radius,
                        Tuuclr, Ruuclr, Tddclr, Tduclr, albedo, debug=False)

    # Protect against new FARMS-DNI which returns (ghi, farms-dni, dni0)
    if isinstance(baseline_ghi, (list, tuple)):
        baseline_ghi = baseline_ghi[0]
    assert isinstance(baseline_ghi, np.ndarray)

    assert tf.is_tensor(tensor_ghi)
    assert np.allclose(baseline_ghi, tensor_ghi.numpy(), rtol=0.001)
