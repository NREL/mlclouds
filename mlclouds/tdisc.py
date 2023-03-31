# -*- coding: utf-8 -*-
"""Tensorflow implementation of the DISC model

Created on Wed Jun 25 13:26:21 2014

@author: gbuster

These are functions adapted from PVL_Python.
There were four main changes from the original code
    1) functions were vectorized
    2) pvl_ephemeris expects UTC time
    3) removed unused result calculations
    4) Water and Pressure were changed to vectors from scalars
"""

import tensorflow as tf
import numpy as np
from farms import SOLAR_CONSTANT, SZA_LIM


def tdisc(ghi, sza, doy, pressure=101325, sza_lim=SZA_LIM):
    """Estimate DNI from GHI using the DISC model.

    *Warning: should only be used for cloudy FARMS data.

    The DISC algorithm converts global horizontal irradiance to direct
    normal irradiance through empirical relationships between the global
    and direct clearness indices.

    Parameters
    ----------
    ghi : np.ndarray
        Global horizontal irradiance in W/m2.
    sza : np.ndarray
        Solar zenith angle in degrees.
    doy : np.ndarray
        Day of year (array of integers).
    pressure : np.ndarray
        Pressure in mbar (same as hPa).
    sza_lim : float | int
        Upper limit for solar zenith angle in degrees. SZA values greater than
        this will be truncated at this value.

    Returns
    -------
    DNI : np.ndarray
        Estimated direct normal irradiance in W/m2.
    """

    if len(doy.shape) < len(sza.shape):
        doy = np.tile(doy.reshape((len(doy), 1)), sza.shape[1])

    ghi = tf.convert_to_tensor(ghi, dtype=tf.float32)
    doy = tf.convert_to_tensor(doy, dtype=tf.float32)
    pressure = tf.convert_to_tensor(pressure, dtype=tf.float32)

    sza_rad = np.radians(sza)
    sza_lim_rad = np.radians(sza_lim)
    sza = tf.convert_to_tensor(sza, dtype=tf.float32)
    sza_rad = tf.convert_to_tensor(sza_rad, dtype=tf.float32)
    sza = tf.where(sza > sza_lim, sza_lim, sza)
    sza_rad = tf.where(sza_rad > sza_lim_rad, sza_lim_rad, sza_rad)

    A = np.zeros_like(ghi)
    B = np.zeros_like(ghi)
    C = np.zeros_like(ghi)

    day_angle = 2. * tf.constant(np.pi) * (doy - 1) / 365

    re_var = (1.00011 + 0.034221 * tf.cos(day_angle)
              + 0.00128 * tf.sin(day_angle)
              + 0.000719 * tf.cos(2. * day_angle)
              + 7.7E-5 * tf.sin(2. * day_angle))

    if len(re_var.shape) < len(sza.shape):
        e = ('re_var has bad shape {} but should be {}!'
             .format(re_var.shape, sza.shape))
        raise ValueError(e)

    I0 = re_var * SOLAR_CONSTANT
    I0h = I0 * np.cos(sza_rad)

    AM = (1. / (tf.cos(sza_rad) + 0.15 * ((93.885 - sza)**-1.253))
          * 100 * pressure / 101325)

    Kt = ghi / I0h
    Kt = tf.where(Kt < 0, 0, Kt)

    A0 = -5.743 + 21.77 * Kt - 27.49 * Kt**2 + 11.56 * Kt**3
    B0 = 41.4 - 118.5 * Kt + 66.05 * Kt**2 + 31.9 * Kt**3
    C0 = -47.01 + 184.2 * Kt - 222. * Kt**2 + 73.81 * Kt**3

    A1 = 0.512 - 1.56 * Kt + 2.286 * Kt**2 - 2.222 * Kt**3
    B1 = 0.37 + 0.962 * Kt
    C1 = -0.28 + 0.932 * Kt - 2.048 * Kt**2

    A = tf.where(Kt > 0.6, A0, A1)
    B = tf.where(Kt > 0.6, B0, B1)
    C = tf.where(Kt > 0.6, C0, C1)

    delKn = A + B * tf.exp(C * AM)

    Knc = (0.866 - 0.122 * AM + 0.0121 * AM**2
           - 0.000653 * AM**3 + 0.000014 * AM**4)

    Kn = Knc - delKn
    DNI = Kn * I0

    DNI = tf.where((sza >= sza_lim) | (ghi < 1) | (DNI < 0), 0, DNI)

    return DNI
