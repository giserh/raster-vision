"""
Execute a sequence of tasks for a run, given a json file with options for that
run. Example usage: `python run.py options.json setup train eval`
"""
import json
from os.path import join, isfile, isdir
import sys
import argparse
from subprocess import call

from .data.preprocess import _makedirs, results_path, get_input_shape
from .train import make_model, train_model
from .eval_run import eval_run

SETUP = 'setup'
TRAIN = 'train'
EVAL = 'eval'


class RunOptions():
    def __init__(self, git_commit=None, model_type=None,
                 nb_labels=None, run_name=None, batch_size=None,
                 samples_per_epoch=None, nb_epoch=None, nb_val_samples=None,
                 nb_prediction_images=None, patience=None, cooldown=None,
                 include_depth=False, kernel_size=None, dataset=None,
                 lr_schedule=None, drop_prob=None, is_big=True):
        # Run `git rev-parse head` to get this.
        self.git_commit = git_commit
        self.model_type = model_type
        self.nb_labels = nb_labels
        self.run_name = run_name
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.nb_epoch = nb_epoch
        self.nb_val_samples = nb_val_samples
        self.nb_prediction_images = nb_prediction_images
        self.patience = patience
        self.cooldown = cooldown
        self.lr_schedule = lr_schedule
        self.dataset = dataset

        if self.model_type == 'conv_logistic':
            self.kernel_size = kernel_size or [1, 1]

        if self.model_type == 'fcn_resnet':
            self.drop_prob = drop_prob or 0.0

        self.is_big = is_big

        self.include_depth = include_depth
        self.set_input_shape()

    def set_input_shape(self, use_big_tiles=False):
        self.input_shape = get_input_shape(self.include_depth, use_big_tiles)


def load_options(file_path):
    options = None
    with open(file_path) as options_file:
        options_dict = json.load(options_file)
        options = RunOptions(**options_dict)

    return options


class Logger(object):
    def __init__(self, run_path):
        self.terminal = sys.stdout
        self.log = open(join(run_path, 'stdout.txt'), 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def setup_run(options, sync_results):
    run_path = join(results_path, options.run_name)
    if not isdir(run_path):
        sync_results(download=True)

    _makedirs(run_path)

    options_json = json.dumps(options.__dict__, sort_keys=True, indent=4)
    options_path = join(run_path, 'options.json')
    with open(options_path, 'w') as options_file:
        options_file.write(options_json)

    sys.stdout = Logger(run_path)


def load_model(options, run_path):
    model = make_model(options)
    model.load_weights(join(run_path, 'model.h5'), by_name=True)
    return model


def train_run(options, run_path, sync_results):
    model_path = join(run_path, 'model.h5')

    if isfile(model_path):
        model = load_model(options, run_path)
        print('Continuing training on {}'.format(model_path))
    else:
        model = make_model(options)
        print('Creating new model.')
    train_model(model, sync_results, options)

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', nargs='?',
                        help='path to the options json file',
                        default='/opt/src/options.json')
    parser.add_argument('tasks', nargs='*', help='list of tasks to perform',
                        default=['setup', 'train', 'eval'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    options = load_options(args.file_path)
    run_path = join(results_path, options.run_name)

    def sync_results(download=False):
        s3_run_path = 's3://otid-data/results/{}'.format(options.run_name)
        if download:
            call(['aws', 's3', 'sync', s3_run_path, run_path])
        else:
            call(['aws', 's3', 'sync', run_path, s3_run_path])

    for task in args.tasks:
        if task == SETUP:
            setup_run(options, sync_results)
        elif task == TRAIN:
            train_run(options, run_path, sync_results)
        elif task == EVAL:
            options.set_input_shape(use_big_tiles=True)
            model = load_model(options, run_path)
            eval_run(model, options)
            sync_results()