"""
Perform a grid search over a phygnn mlclouds model.
"""
import argparse
import pandas as pd

from getpass import getuser
from itertools import product
from os import makedirs
from os.path import join

from configobj import ConfigObj, ConfigspecError, flatten_errors
from rex.utilities.hpc import SLURM
from validate import Validator


DATA_FILES = {'east': {2016: '2016_east_adj/mlclouds_surfrad_2016.h5',
                       2017: '2017_east_adj/mlclouds_surfrad_2017.h5',
                       2018: '2018_east_adj/mlclouds_surfrad_2018_adj.h5',
                       2019: '2019_east_adj/mlclouds_surfrad_2019_adj.h5'},
              'west': {2016: '2016_west_adj/mlclouds_surfrad_2016.h5',
                       2017: '2017_west_adj/mlclouds_surfrad_2017.h5',
                       2018: '2018_west_adj/mlclouds_surfrad_2018.h5',
                       2019: '2019_west_adj/mlclouds_surfrad_2019.h5'}
              }

CONFIG = {
    "surfrad_window_minutes": 15,
    "features": ["solar_zenith_angle",
                 "cloud_type",
                 "refl_0_65um_nom",
                 "refl_0_65um_nom_stddev_3x3",
                 "refl_3_75um_nom",
                 "temp_3_75um_nom",
                 "temp_11_0um_nom",
                 "temp_11_0um_nom_stddev_3x3",
                 "cloud_probability",
                 "cloud_fraction",
                 "air_temperature",
                 "dew_point",
                 "relative_humidity",
                 "total_precipitable_water",
                 "surface_albedo"
                 ],
    "y_labels": ["cld_opd_dcomp", "cld_reff_dcomp"],
    "hidden_layers": [{"units": 64, "activation": "relu", "name": "relu1",
                       "dropout": 0.01},
                      {"units": 64, "activation": "relu", "name": "relu2",
                       "dropout": 0.01},
                      {"units": 64, "activation": "relu", "name": "relu3",
                       "dropout": 0.01}
                      ],
    "phygnn_seed": 0,
    "metric": "relative_mae",
    "learning_rate": 1e-3,
    "n_batch": 16,
    "epochs_a": 10,
    "epochs_b": 10,
    "loss_weights_a": [1, 0],
    "loss_weights_b": [0.5, 0.5],
    "p_kwargs": {"loss_terms": ["mae_ghi", "mae_dni", "mbe_ghi", "mbe_dni"]},
    "p_fun": "p_fun_all_sky",
    "clean_training_data_kwargs": {"filter_clear": False,
                                   "nan_option": "interp"},
    "one_hot_categories": {"flag": ["clear", "ice_cloud", "water_cloud",
                                    "bad_cloud"]}
}


class GridSearcher(object):
    """
    Perform a grid search over provided model hyperparmaters.
    """
    def __init__(self, output_ws, exe_fpath,
                 data_root='/lustre/eaglefs/projects/mlclouds/data_surfrad_9/',
                 conda_env='mlclouds', number_hidden_layers=(3, ),
                 number_hidden_nodes=(64, ), dropouts=(0.01, ),
                 learning_rates=(0.001, ), loss_weights_b=([0.5, 0.5], ),
                 test_fractions=(0.2, ), epochs_a=(10, ), epochs_b=(10, ),
                 base_config=CONFIG):
        """
        Parameters
        ----------
        output_ws: str
            Filepath to folder used for config file and output files storage.
            Must have write access.
        exe_fpath: str
            Filepath to 'run_mlclouds.py'.
        data_root: str
            Filepath to surfrad data root. Defaults to
            '/lustre/eaglefs/projects/mlclouds/data_surfrad_9/'.
        conda_env: str
            Anaconda environment for HPC jobs. Defaults to mlclouds.
        number_hidden_layers: list of int
            Number of fully-connected, relu activated model layers to compile.
            <dropouts> values are applied. Defaults to (3, ).
        number_hidden_nodes: list of int
            Layer depth for each hidden layer in <number_hidden_layers>.
            Defaults to (64, ).
        dropouts: list of float
            Droput rates applied to each layer. Should be between 0 and 1.
            Defaults to (0.01, ).
        learning_rates: list of float
            Model learning rates. Defaults to (0.001, ).
        loss_weights_b: list of list of float
            Loss function weights applied in second round of training.
            First weight applies to MSE, second to physics loss function.
            Should sum to 1. Defaults to ([0.5, 0.5], ).
        test_fractions: list of float
            Fraction of training samples to be withheld for testing. Should
            be between 0 and 1. Defaults to (0.2, ).
        epochs_a: list of int
            Number of epochs to train without physics loss function applied.
            Defaults to (10, 0).
        epochs_b: list of int
            Number of epochs to train with physcs loss function applied.
            Defaults to (10, ).
        base_config:
            Base configuration for model. Defaults to:

                {"surfrad_window_minutes": 15,
                 "features": ["solar_zenith_angle",
                              "cloud_type",
                              "refl_0_65um_nom",
                              "refl_0_65um_nom_stddev_3x3",
                              "refl_3_75um_nom",
                              "temp_3_75um_nom",
                              "temp_11_0um_nom",
                              "temp_11_0um_nom_stddev_3x3",
                              "cloud_probability",
                              "cloud_fraction",
                              "air_temperature",
                              "dew_point",
                              "relative_humidity",
                              "total_precipitable_water",
                              "surface_albedo"
                              ],
                 "y_labels": ["cld_opd_dcomp", "cld_reff_dcomp"],
                 "hidden_layers": [{"units": 64, "activation": "relu",
                                    "name": "relu1", "dropout": 0.01},
                                   {"units": 64, "activation": "relu",
                                    "name": "relu2", "dropout": 0.01},
                                   {"units": 64, "activation": "relu",
                                    "name": "relu3", "dropout": 0.01}
                                   ],
                 "phygnn_seed": 0,
                 "metric": "relative_mae",
                 "learning_rate": 1e-3,
                 "n_batch": 16,
                 "epochs_a": 10,
                 "epochs_b": 10,
                 "loss_weights_a": [1, 0],
                 "loss_weights_b": [0.5, 0.5],
                 "p_kwargs": {"loss_terms": ["mae_ghi", "mae_dni", "mbe_ghi",
                                             "mbe_dni"]},
                 "p_fun": "p_fun_all_sky",
                 "clean_training_data_kwargs": {"filter_clear": False,
                                                "nan_option": "interp"},
                 "one_hot_categories": {"flag": ["clear", "ice_cloud",
                                                 "water_cloud", "bad_cloud"]}
                }
       """
        self._output_ws = None
        self.output_ws = output_ws

        self.jobs = list(product(number_hidden_layers, number_hidden_nodes,
                                 dropouts, learning_rates, loss_weights_b,
                                 test_fractions, epochs_a, epochs_b))

        self.results = pd.DataFrame(self.jobs, columns=['number_hidden_layers',
                                                        'number_hidden_nodes',
                                                        'dropout',
                                                        'learning_rate',
                                                        'loss_weights_b',
                                                        'test_fraction',
                                                        'epochs_a',
                                                        'epochs_b'])
        for var in ['elapsed_time', 'training_loss', 'validation_loss']:
            self.results[var] = [None, ] * len(self.jobs)

        self.base_config = base_config

        self.conda_env = conda_env

        files = []
        for region in DATA_FILES:
            for year in DATA_FILES[region]:
                files.append(join(data_root, DATA_FILES[region][year]))
        self.base_config['files'] = files

        self.slurm = SLURM()

        self.job_ids = []
        self.job_stdout = []

        self.exe_fpath = exe_fpath
        self.config_fpath = join(self.output_ws, 'clds_opt_{id}.ini')
        self.stats_fpath = join(self.output_ws, 'clds_opt_{id}_stats.csv')
        self.log_fpath = join(self.output_ws, 'clds_opt_{id}.log')
        self.history_fpath = join(self.output_ws,
                                  'clds_opt_{id}_training_history.csv')
        self.model_fpath = join(self.output_ws, 'clds_opt_{id}.pkl')

    @property
    def output_ws(self):
        """
        Filepath of folder to contain output files.

        Returns
        -------
        output_ws: str
             Filepath of folder to contain output files.
        """
        return self._output_ws

    @output_ws.setter
    def output_ws(self, value):
        try:
            makedirs(value, exist_ok=True)
        except Exception as e:
            raise e
        else:
            self._output_ws = value

    def start_job(self, number_hidden_layers, number_hidden_nodes, dropout,
                  learning_rate, loss_weights_b, test_fraction, epochs_a,
                  epochs_b, id=0):
        """
        Start a single HPC task for a single model run via run_mlclouds.py.

        Parameters
        ----------
        number_hidden_layers: int
            Number of fully-connected, relu activated model layers to compile.
            <dropout> value is applied.
        number_hidden_nodes: int
            Layer depth for each hidden layer in <number_hidden_layers>.
        dropout: float
            Droput rate applied to each layer. Should be between 0 and 1.
        learning_rate: float
            Model learning rate.
        loss_weights_b: list of float
            Loss function weights applied in second round of training. First
            weight applies to MSE, second to physics loss function. Should sum
            to 1.
        test_fraction: float
            Fraction of training samples to be withheld for testing. Should be
            between 0 and 1.
        epochs_a: int
            Number of epochs to train without physics loss function applied.
        epochs_b: int
            Number of epochs to train with physcs loss function applied.
        id: int
            Run ID number. Defaults to 0.
        """

        config = ConfigObj(self.base_config)

        hidden_layers = [{"units": number_hidden_nodes, "activation": "relu",
                          "dropout": dropout}] * number_hidden_layers

        config.update(
            {'hidden_layers': hidden_layers,
             'learning_rate': learning_rate,
             'loss_weights_b': loss_weights_b,
             'epochs_a': epochs_a,
             'epochs_b': epochs_b
             }
        )

        config.filename = self.config_fpath.format(id=id)
        config.write()

        cmd = f'python {self.exe_fpath} {self.config_fpath.format(id=id)} '\
              f'{test_fraction} ' \
              f'--stats_file={self.stats_fpath.format(id=id)} ' \
              f'--log={self.log_fpath.format(id=id)} ' \
              f'--log_level=DEBUG ' \
              f'--training_history={self.history_fpath.format(id=id)} ' \
              f'--model_path={self.model_fpath.format(id=id)}'

        jobid, stdout = self.slurm.sbatch(cmd, alloc='mlclouds', walltime=1,
                                          memory=None, feature=None,
                                          name=f'clds_opt_{id}',
                                          stdout_path=self.output_ws,
                                          keep_sh=False,
                                          conda_env=self.conda_env,
                                          module=None,
                                          module_root=None)

        self.job_ids.append(jobid)
        self.job_stdout.append(stdout)

    def run_grid_search(self):
        """
        Start an HPC job for each job in self.jobs.
        """
        for i, job in enumerate(self.jobs):
            number_hidden_layers, number_hidden_nodes, dropout, \
                learning_rate, loss_weights_b, test_fraction, epochs_a, \
                epochs_b = job

            self.start_job(number_hidden_layers, number_hidden_nodes, dropout,
                           learning_rate, loss_weights_b, test_fraction,
                           epochs_a, epochs_b, id=i)

    def jobs_status(self):
        """
        Query SLURM queue for active jobs for current user.

        Returns
        -------
        status: str
            SLURM status for all active jobs for current user.
        """
        return self.slurm.query_queue()

    def collect_results(self):
        """
        Collect training metrics for each job in self.jobs. Assumes all jobs
        have completed successfully.

        Returns
        -------
        results: DataFrame
            Pandas DataFrame with columns: epoch, elapsed_time, training_loss,
            validation_loss, number_hidden_layers, number_hidden_nodes,
            dropout, learning_rate, loss_weights_b, test_fraction, epochs_a,
            epochs_b.
        """
        for i in range(len(self.jobs)):
            try:
                df = pd.read_csv(self.history_fpath.format(id=i)).iloc[[-1]]
            except IOError:
                continue
            else:
                idx = self.results.index == i
                for var in ['elapsed_time', 'training_loss',
                            'validation_loss']:
                    self.results.loc[idx, var] = df[var].values

        return self.results


if __name__ == '__main__':
    user = getuser()

    parser = argparse.ArgumentParser(description='Conduct a hyperparameter '
                                                 'grid search over a '
                                                 'mlclouds Phygnn model')
    parser.add_argument('config', type=str,
                        help='Grid search configuration file')
    parser.add_argument('--conda_env', type=str, default='mlclouds',
                        help='Anaconda environment for HPC jobs. Defaults'
                        'to mlclouds')
    parser.add_argument('--output_ws', type=str,
                        default=f'/scratch/{user}/mlclouds/optimization/',
                        help='Output folder for stats, training history,'
                        ' etc. Defaults to /scratch/{user}/mlclouds/'
                        'optimization/.')
    parser.add_argument('--exe_fpath', type=str,
                        default='~/src/mlclouds/mlclouds/scripts/train.py',
                        help='File path to train.py. Defaults to'
                        '~/src/mlclouds/mlclouds/scripts/train.py')
    parser.add_argument('--data_root', type=str,
                        default='/lustre/eaprojects/mlclouds/data_surfrad_9/',
                        help='Surfrad data root directory. Defaults to'
                        '/lustre/eaprojects/mlclouds/data_surfrad_9/')
    parser.add_argument('--collect_results', action='store_true',
                        help='Collect results instead of run jobs.'
                        'Saved as {output_ws}/results.csv')

    args = parser.parse_args()

    validator = Validator()
    config = ConfigObj(args.config, configspec='gridsearch.spec',
                       stringify=True)
    validation = config.validate(validator, preserve_errors=True)

    if validation is not True:
        msg = ''
        for entry in flatten_errors(config, validation):
            section_list, key, error = entry
            if key is not None:
                if error is False:
                    msg += f'{key}: missing\n'
                else:
                    msg += f'{key}: {error}\n'
        raise ConfigspecError(msg)

    loss_weights_b = []
    for loss_weight in config['loss_weights_b']:
        loss_weights_b.append([round(1.0 - loss_weight, 2), loss_weight])

    kvals = {
        'conda_env': args.conda_env,
        'data_root': args.data_root,
        'exe_fpath': args.exe_fpath,
        'output_ws': args.output_ws,
        'number_hidden_layers': config['number_hidden_layers'],
        'number_hidden_nodes': config['number_hidden_nodes'],
        'dropouts': config['dropouts'],
        'learning_rates': config['learning_rates'],
        'loss_weights_b': config['loss_weights_b'],
        'test_fractions': config['test_fractions'],
        'epochs_a': config['epochs_a'],
        'epochs_b': config['epochs_b']
    }

    GS = GridSearcher(**kvals)

    if not args.collect_results:
        GS.run_grid_search()
    else:
        GS.collect_results().to_csv(join(args.output_ws, 'results.csv'),
                                    index=True)
