"""
Automatic cross validation of PhyGNN models predicting opd and reff

Mike Bannister 7/2020
Based on code by Grant Buster
"""
import numpy as np
import logging

import tensorflow as tf

from mlclouds.tdisc import tdisc
from mlclouds.tfarms import tfarms
from mlclouds.tdhi import t_calc_dhi


logger = logging.getLogger(__name__)


def p_fun_dummy(model, y_true, y_predicted, p, labels=None, loss_terms=None):
    """Dummy loss function to disable pfun"""
    # pylint: disable-msg=W0613
    return tf.constant(0.0, dtype=tf.float32)


def p_fun_all_sky(model, y_true, y_predicted, p, labels=None,
                  loss_terms=('mae_ghi', )):
    """Physics loss function """
    # pylint: disable-msg=W0613
    n = len(y_true)
    tau = tf.expand_dims(y_predicted[:, 0], axis=1)
    cld_reff = tf.expand_dims(y_predicted[:, 1], axis=1)

    tau = tf.where(tau < 0, 0.0001, tau)
    tau = tf.where(tau > 160, 160, tau)
    cld_reff = tf.where(cld_reff < 0, 0.0001, cld_reff)
    cld_reff = tf.where(cld_reff > 160, 160, cld_reff)

    cloud_type = p[:, labels.index('cloud_type')
                   ].astype(np.float32).reshape((n, 1))
    solar_zenith_angle = p[:, labels.index('solar_zenith_angle')
                           ].astype(np.float32).reshape((n, 1))
    doy = p[:, labels.index('doy')].astype(np.float32).reshape((n, 1))
    radius = p[:, labels.index('radius')].astype(np.float32).reshape((n, 1))
    Tuuclr = p[:, labels.index('Tuuclr')].astype(np.float32).reshape((n, 1))
    Ruuclr = p[:, labels.index('Ruuclr')].astype(np.float32).reshape((n, 1))
    Tddclr = p[:, labels.index('Tddclr')].astype(np.float32).reshape((n, 1))
    Tduclr = p[:, labels.index('Tduclr')].astype(np.float32).reshape((n, 1))
    albedo = p[:, labels.index('surface_albedo')
               ].astype(np.float32).reshape((n, 1))
    pressure = p[:, labels.index('surface_pressure')
                 ].astype(np.float32).reshape((n, 1))

    cs_dni = p[:, labels.index('clearsky_dni')]
    cs_ghi = p[:, labels.index('clearsky_ghi')]
    cs_dni = cs_dni.astype(np.float32).reshape((n, 1))
    cs_ghi = cs_ghi.astype(np.float32).reshape((n, 1))
    cs_dni = tf.convert_to_tensor(cs_dni, dtype=tf.float32)
    cs_ghi = tf.convert_to_tensor(cs_ghi, dtype=tf.float32)

    i_dhi = labels.index('surfrad_dhi')
    i_dni = labels.index('surfrad_dni')
    i_ghi = labels.index('surfrad_ghi')
    dhi_ground = p[:, i_dhi].astype(np.float32).reshape((n, 1))
    dni_ground = p[:, i_dni].astype(np.float32).reshape((n, 1))
    ghi_ground = p[:, i_ghi].astype(np.float32).reshape((n, 1))
    dhi_ground = tf.convert_to_tensor(dhi_ground, dtype=tf.float32)
    dni_ground = tf.convert_to_tensor(dni_ground, dtype=tf.float32)
    ghi_ground = tf.convert_to_tensor(ghi_ground, dtype=tf.float32)

    ghi_predicted = tfarms(tau, cloud_type, cld_reff, solar_zenith_angle,
                           radius, Tuuclr, Ruuclr, Tddclr, Tduclr, albedo)
    dni_predicted = tdisc(ghi_predicted, solar_zenith_angle, doy,
                          pressure=pressure)

    dni_predicted = tf.where(tau == 0.0001, cs_dni, dni_predicted)
    ghi_predicted = tf.where(tau == 0.0001, cs_ghi, ghi_predicted)
    dhi_predicted, dni_predicted = t_calc_dhi(dni_predicted, ghi_predicted,
                                              solar_zenith_angle)

    err_ghi = ghi_predicted - ghi_ground
    err_ghi = tf.boolean_mask(err_ghi, ~tf.math.is_nan(err_ghi))
    err_ghi = tf.boolean_mask(err_ghi, tf.math.is_finite(err_ghi))

    err_dni = dni_predicted - dni_ground
    err_dni = tf.boolean_mask(err_dni, ~tf.math.is_nan(err_dni))
    err_dni = tf.boolean_mask(err_dni, tf.math.is_finite(err_dni))

    err_dhi = dhi_predicted - dhi_ground
    err_dhi = tf.boolean_mask(err_dhi, ~tf.math.is_nan(err_dhi))
    err_dhi = tf.boolean_mask(err_dhi, tf.math.is_finite(err_dhi))

    terms = {}
    terms['mae_ghi'] = (tf.reduce_mean(tf.abs(err_ghi))
                        / tf.reduce_mean(ghi_ground))
    terms['mae_dni'] = (tf.reduce_mean(tf.abs(err_dni))
                        / tf.reduce_mean(dni_ground))
    terms['mae_dhi'] = (tf.reduce_mean(tf.abs(err_dhi))
                        / tf.reduce_mean(dhi_ground))
    terms['mbe_ghi'] = (tf.abs(tf.reduce_mean(err_ghi))
                        / tf.reduce_mean(ghi_ground))
    terms['mbe_dni'] = (tf.abs(tf.reduce_mean(err_dni))
                        / tf.reduce_mean(dni_ground))
    terms['mbe_dhi'] = (tf.abs(tf.reduce_mean(err_dhi))
                        / tf.reduce_mean(dhi_ground))
    terms['rmse_ghi'] = (tf.sqrt(tf.reduce_mean(tf.square(err_ghi)))
                         / tf.reduce_mean(ghi_ground))
    terms['rmse_dni'] = (tf.sqrt(tf.reduce_mean(tf.square(err_dni)))
                         / tf.reduce_mean(dni_ground))
    terms['rmse_dhi'] = (tf.sqrt(tf.reduce_sum(tf.square(err_dhi)))
                         / tf.reduce_mean(dhi_ground))

    p_loss = sum([terms[x] for x in loss_terms])
    return p_loss
