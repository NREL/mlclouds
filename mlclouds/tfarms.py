"""
Tensor-based FARMS model for radiative tranfer predictions by TensorFlow DNNs

Created on Fri June 1 2015
FAST Model
Adapted from Yu Xie IDL Fast Radiative Transfer Model
@author: Grant Buster

This Fast All-sky Radiation Model for Solar applications (FARMS) was developed
by Yu Xie (Yu.Xie@nrel.gov). Please contact him for more information.

Literature
----------
[1] Yu Xie, Manajit Sengupta, Jimy Dudhia, "A Fast All-sky Radiation Model
    for Solar applications (FARMS): Algorithm and performance evaluation",
    Solar Energy, Volume 135, 2016, Pages 435-445, ISSN 0038-092X,
    https://doi.org/10.1016/j.solener.2016.06.003.
    (http://www.sciencedirect.com/science/article/pii/S0038092X16301827)
"""
import tensorflow as tf
import numpy as np
import collections
from farms import WATER_TYPES, SOLAR_CONSTANT
from farms.utilities import check_range
from phygnn.utilities import tf_isin, tf_log10


def water_phase(tau, De, solar_zenith_angle):
    """Get cloudy Tducld and Ruucld for the water phase."""

    # 12a from [1]
    Ptau = (2.8850 + 0.002 * (De - 60.0)) * solar_zenith_angle - 0.007347

    # 12b from [1]
    PDHI = (0.7846 * (1.0 + 0.0002 * (De - 60.0))
            * (solar_zenith_angle**0.1605))

    # 12c from [1]
    delta = (-0.644531 * solar_zenith_angle + 1.20117 + 0.129807
             / solar_zenith_angle - 0.00121096
             / (solar_zenith_angle * solar_zenith_angle) + 1.52587e-07
             / (solar_zenith_angle * solar_zenith_angle * solar_zenith_angle))

    # part of 12d from [1]
    y = 0.012 * (tau - Ptau) * solar_zenith_angle

    # 11 from [1]
    Tducld = ((1.0 + tf.sinh(y)) * PDHI
              * tf.exp(-((tf_log10(tau) - tf_log10(Ptau))**2) / delta))

    # 14a from [1]
    Ruucld0 = 0.107359 * tau
    Ruucld1 = 1.03 - tf.exp(-(0.5 + tf_log10(tau))
                            * (0.5 + tf_log10(tau)) / 3.105)
    Ruucld = tf.where(tau < 1.0, Ruucld0, Ruucld1)

    return Tducld, Ruucld


def ice_phase(tau, De, solar_zenith_angle):
    """Get cloudy Tducld and Ruucld for the ice phase."""

    # 13a from [1]
    Ptau0 = 2.8487 * solar_zenith_angle - 0.0029
    Ptau1 = (2.8355 + (100.0 - De) * 0.006) * solar_zenith_angle - 0.00612
    Ptau = tf.where(De <= 26.0, Ptau0, Ptau1)

    # 13b from [1]
    PDHI = 0.756 * solar_zenith_angle**0.0883

    # 13c from [1]
    delta = (-0.0549531 * solar_zenith_angle + 0.617632
             + (0.17876 / solar_zenith_angle)
             - (0.002174 / solar_zenith_angle ** 2))

    # part of 13c from [1]
    y = 0.01 * (tau - Ptau) * solar_zenith_angle

    # 11 from [1]
    Tducld = ((1.0 + tf.sinh(y)) * PDHI
              * tf.exp(-((tf_log10(tau) - tf_log10(Ptau))**2) / delta))

    # 14b from [1]
    Ruucld0 = 0.094039 * tau
    Ruucld1 = 1.02 - tf.exp(-(0.5 + tf_log10(tau))
                            * (0.5 + tf_log10(tau)) / 3.25)
    Ruucld = tf.where(tau < 1.0, Ruucld0, Ruucld1)

    return Tducld, Ruucld


def tfarms(tau, cloud_type, cloud_effective_radius, solar_zenith_angle,
           radius, Tuuclr, Ruuclr, Tddclr, Tduclr, albedo, debug=False):
    """Fast All-sky Radiation Model for Solar applications (FARMS).

    Literature
    ----------
    [1] Yu Xie, Manajit Sengupta, Jimy Dudhia, "A Fast All-sky Radiation Model
        for Solar applications (FARMS): Algorithm and performance evaluation",
        Solar Energy, Volume 135, 2016, Pages 435-445, ISSN 0038-092X,
        https://doi.org/10.1016/j.solener.2016.06.003.
        (http://www.sciencedirect.com/science/article/pii/S0038092X16301827)

    Variables
    ---------
    F0
        Radiative flux at top of atmosphere
    Fd
        Direct solar flux in the downwelling direction at the surface
        (eq 2a from [1])
    De
        Effective cloud particle size (diameter).


    Parameters
    ----------
    tau : np.ndarray
        Cloud optical thickness (cld_opd_dcomp) (unitless).
    cloud_type : np.ndarray
        Integer values representing different cloud types
        https://github.nrel.gov/PXS/pxs/wiki/Cloud-Classification
    cloud_effective_radius : np.ndarray
        Cloud effective particle radius (cld_reff_dcomp) (micron).
    solar_zenith_angle : np.ndarray
        Solar zenith angle (degrees). Must represent the average value over the
        integration period (e.g. hourly) under scrutiny.
    radius : np.ndarray
        Sun-earth radius vector, varies between 1.017 in July and
        0.983 in January.
    Tuuclr : np.ndarray
        Transmittance of the clear-sky atmosphere for diffuse incident and
        diffuse outgoing fluxes (uu).
        ***Calculated from multiple REST2 runs at different solar angles.
        Average of Tddclr w different solar angles (see eq 5 from [1]).
    Ruuclr : np.ndarray
        Calculated in REST2. Aerosol reflectance for diffuse fluxes.
    Tddclr : np.ndarray
        Calculated in REST2. Transmittance of the clear-sky atmosphere for
        direct incident and direct outgoing fluxes (dd).
        Tddclr = dni / etdirn
    Tduclr : np.ndarray
        Calculated in REST2. Transmittance of the clear-sky atmosphere for
        direct incident and diffuse outgoing fluxes (du).
        Tduclr = dhi / (etdirn * cosz)
    albedo : np.ndarray
        Ground albedo.
    debug : bool
        Flag to output additional transmission/reflectance variables.

    Returns
    -------
    ghi : np.ndarray
        FARMS GHI values (this is the only output if debug is False).
    fast_data : collections.namedtuple
        Additional debugging variables if debug is True.
        Named tuple with irradiance data. Attributes:
            ghi : global horizontal irradiance (w/m2)
            dni : direct normal irradiance (w/m2)
            dhi : diffuse horizontal irradiance (w/m2)
    """
    # disable divide by zero warnings
    np.seterr(divide='ignore')

    check_range(Tddclr, 'Tddclr')
    check_range(Tduclr, 'Tduclr')
    check_range(Ruuclr, 'Ruuclr')
    check_range(Tuuclr, 'Tuuclr')

    # do not allow for negative cld optical depth
    tau = tf.where(tau < 0, 0.001, tau)
    tau = tf.where(tau > 160, 160, tau)
    cloud_effective_radius = tf.where(cloud_effective_radius < 0, 0.001,
                                      cloud_effective_radius)
    cloud_effective_radius = tf.where(cloud_effective_radius > 160, 160,
                                      cloud_effective_radius)

    F0 = SOLAR_CONSTANT / (radius * radius)
    solar_zenith_angle = np.cos(np.radians(solar_zenith_angle))
    De = 2.0 * cloud_effective_radius

    tau = tf.convert_to_tensor(tau, dtype=tf.float32)
    F0 = tf.convert_to_tensor(F0, dtype=tf.float32)
    De = tf.convert_to_tensor(De, dtype=tf.float32)
    cloud_effective_radius = tf.convert_to_tensor(cloud_effective_radius,
                                                  dtype=tf.float32)
    solar_zenith_angle = tf.convert_to_tensor(solar_zenith_angle,
                                              dtype=tf.float32)
    radius = tf.convert_to_tensor(radius, dtype=tf.float32)
    Tuuclr = tf.convert_to_tensor(Tuuclr, dtype=tf.float32)
    Ruuclr = tf.convert_to_tensor(Ruuclr, dtype=tf.float32)
    Tddclr = tf.convert_to_tensor(Tddclr, dtype=tf.float32)
    Tduclr = tf.convert_to_tensor(Tduclr, dtype=tf.float32)
    albedo = tf.convert_to_tensor(albedo, dtype=tf.float32)

    phase1 = tf_isin(cloud_type, WATER_TYPES)

    Tducld_p1, Ruucld_p1 = water_phase(tau, De, solar_zenith_angle)
    Tducld_p2, Ruucld_p2 = ice_phase(tau, De, solar_zenith_angle)

    Tducld = tf.where(phase1, Tducld_p1, Tducld_p2)
    Ruucld = tf.where(phase1, Ruucld_p1, Ruucld_p2)

    # eq 8 from [1]
    Tddcld = tf.exp(-tau / solar_zenith_angle)

    Fd = solar_zenith_angle * F0 * Tddcld * Tddclr  # eq 2a from [1]
    F1 = solar_zenith_angle * F0 * (Tddcld * (Tddclr + Tduclr)
                                    + Tducld * Tuuclr)  # eq 3 from [1]

    # ghi eqn 6 from [1]
    ghi = F1 / (1.0 - albedo * (Ruuclr + Ruucld * Tuuclr * Tuuclr))
    dni = Fd / solar_zenith_angle  # eq 2b from [1]
    dhi = ghi - Fd  # eq 7 from [1]

    if debug:
        # Return NaN if clear-sky, else return cloudy sky data
        fast_data = collections.namedtuple('fast_data', ['ghi', 'dni', 'dhi',
                                                         'Tddcld', 'Tducld',
                                                         'Ruucld'])
        fast_data.Tddcld = Tddcld
        fast_data.Tducld = Tducld
        fast_data.Ruucld = Ruucld
        fast_data.ghi = ghi
        fast_data.dni = dni
        fast_data.dhi = dhi

        return fast_data
    else:
        # return only GHI
        return ghi
