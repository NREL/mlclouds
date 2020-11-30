"""
Train a phygnn mlclouds model.
"""
import argparse

from datetime import datetime

from configobj import ConfigObj
from mlclouds.autoxval import TrainTest
from rex.utilities.loggers import init_logger


if __name__ == '__main__':

    now = datetime.now().strftime('%Y%m%d_%H%M%S')

    parser = argparse.ArgumentParser(description='Execute a Phygnn model')
    parser.add_argument('config', type=str, help='config filepath')
    parser.add_argument('test_fraction', type=float,
                        help='percent test fraction')
    parser.add_argument('--stats_file', type=str, help='output stats filepath',
                        default='{}_stats.csv'.format(now))
    parser.add_argument('--log', type=str, help="log filepath",
                        default='{}.log'.format(now))
    parser.add_argument('--log_level', type=str, help='logging level',
                        default='INFO')
    parser.add_argument('--training_history', type=str,
                        help='training history filepath',
                        default='{}_training_history.csv'.format(now))
    parser.add_argument('--model_path', type=str, help='save model filepath',
                        default='{}_model.pkl')

    args = parser.parse_args()

    init_logger(logger_name='phygnn', log_level=args.log_level,
                log_file=args.log)

    config = ConfigObj(args.config)
    test_fraction = args.test_fraction
    stats_file = args.stats_file
    training_history_fpath = args.training_history
    model_fpath = args.model_path
    files = config.as_list('files')

    tt = TrainTest(files, config=config, test_fraction=test_fraction,
                   stats_file=stats_file)

    tt.trainer.model.history.to_csv(training_history_fpath)
    tt.trainer.model.save(model_fpath)
