"""MLClouds library."""

import os

from ._version import __version__

MLCLOUDSDIR = os.path.dirname(os.path.realpath(__file__))
TESTDATADIR = os.path.join(os.path.dirname(MLCLOUDSDIR), 'tests', 'data')

LEG_DIR = os.path.join(MLCLOUDSDIR, 'model/legacy')
CTYPE_DIR = os.path.join(MLCLOUDSDIR, 'model/cloud_type')
CPROP_DIR = os.path.join(MLCLOUDSDIR, 'model/cloud_properties')
COMBINED_DIR = os.path.join(MLCLOUDSDIR, 'model/type_plus_properties')
CONFIG_FPATH = os.path.join(LEG_DIR, 'config.json')

CPROP_MODEL_FPATH = os.path.join(CPROP_DIR, 'outputs/mlclouds_model.pkl')
CTYPE_MODEL_FPATH = os.path.join(CTYPE_DIR, 'outputs/mlclouds_model.pkl')
COMBINED_MODEL_FPATH = os.path.join(COMBINED_DIR, 'outputs/mlclouds_model.pkl')
LEG_MODEL_FPATH = os.path.join(LEG_DIR, 'outputs/mlclouds_model.pkl')

MODEL_FPATH = {
    'cloud_type_model_path': CTYPE_MODEL_FPATH,
    'cloud_prop_model_path': CPROP_MODEL_FPATH,
}
