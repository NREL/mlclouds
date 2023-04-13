#, -*- coding: utf-8 -*-
"""MLClouds library."""
import os

MLCLOUDSDIR = os.path.dirname(os.path.realpath(__file__))
TESTDATADIR = os.path.join(os.path.dirname(MLCLOUDSDIR), 'tests', 'data')
PROD_DIR = os.path.join(MLCLOUDSDIR, 'model/production_model')
CONFIG_FPATH = os.path.join(PROD_DIR, 'config.json')
MODEL_FPATH = os.path.join(PROD_DIR, 'outputs/mlclouds_model.pkl')
