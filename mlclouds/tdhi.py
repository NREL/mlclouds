# -*- coding: utf-8 -*-
"""Tensorflow impelemtation of the DHI calculation

Created on Aug 3rd 2020

@author: gbuster
"""
import tensorflow as tf


def t_calc_dhi(dni, ghi, sza):
    """Calculate the diffuse horizontal irradiance and correct the direct.

    Parameters
    ----------
    dni : np.ndarray
        Direct normal irradiance.
    ghi : np.ndarray
        Global horizontal irradiance.
    sza : np.ndarray
        Solar zenith angle (degrees).

    Returns
    -------
    dhi : np.ndarray
        Diffuse horizontal irradiance. This is ensured to be non-negative.
    dni : np.ndarray
        Direct normal irradiance. This is set to zero where dhi < 0
    """

    dni = tf.convert_to_tensor(dni, dtype=tf.float32)
    ghi = tf.convert_to_tensor(ghi, dtype=tf.float32)
    sza = tf.convert_to_tensor(sza, dtype=tf.float32)

    dhi = ghi - dni * tf.cos(sza)

    dni = tf.where(dhi < 0.0, 0.0, ghi)
    dhi = tf.where(dhi < 0.0, ghi, dhi)
    return dhi, dni
