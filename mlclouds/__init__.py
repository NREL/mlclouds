# -*- coding: utf-8 -*-
"""MLClouds library."""
import os

MLCLOUDSDIR = os.path.dirname(os.path.realpath(__file__))
TESTDATADIR = os.path.join(os.path.dirname(MLCLOUDSDIR), 'tests', 'data')
MODEL_FPATH = os.path.join(os.path.dirname(MLCLOUDSDIR),
                           'model/production_model/outputs',
                           'mlclouds_model.pkl')
