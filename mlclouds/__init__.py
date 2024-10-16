"""MLClouds library."""

import os

from ._version import __version__

MLCLOUDSDIR = os.path.dirname(os.path.realpath(__file__))
TESTDATADIR = os.path.join(os.path.dirname(MLCLOUDSDIR), 'tests', 'data')

CTYPE_DIR = os.path.join(MLCLOUDSDIR, 'model/cloud_type')
CTYPE_CONFIG_FPATH = os.path.join(CTYPE_DIR, 'config.json')
CTYPE_MODEL_FPATH = os.path.join(CTYPE_DIR, 'outputs/mlclouds_model.pkl')

CPROP_DIR = os.path.join(MLCLOUDSDIR, 'model/cloud_properties')
CPROP_CONFIG_FPATH = os.path.join(CPROP_DIR, 'config.json')
CPROP_MODEL_FPATH = os.path.join(CPROP_DIR, 'outputs/mlclouds_model.pkl')

MODEL_FPATH = {
    'cloud_type_model_path': CTYPE_MODEL_FPATH,
    'cloud_prop_model_path': CPROP_MODEL_FPATH,
}

LEG_DIR = os.path.join(MLCLOUDSDIR, 'model/legacy')
LEG_CONFIG_FPATH = os.path.join(LEG_DIR, 'config.json')
LEG_MODEL_FPATH = os.path.join(LEG_DIR, 'outputs/mlclouds_model.pkl')
