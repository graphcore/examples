# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

# coding=utf-8
import argparse
import json
import os

from experiments.generative import Generative as Experiment


argp = argparse.ArgumentParser(prog='Contrastive Divergence VAE',
                               description='Code to train and evaluate an MCMC/VAE hybrid.')
argp.add_argument('-r', '--results-dir',
                  help='Path to directory in which to store results.',
                  type=str,
                  default='results/')
argp.add_argument('-R', '--repeats',
                  help='How many runs of the experiment to do.',
                  type=int,
                  default=1)
argp.add_argument('-b', '--micro-batch-size',
                  help='Batch size to use during training. This will override the config file',
                  type=int,
                  default=100)
argp.add_argument('-n', '--run-name',
                  help='The name given to this run. Results will be stored in a subfolder with this title.',
                  type=str,
                  default=None)
argp.add_argument('-v', '--validation',
                  help='Flag - whether to run model validation. 10,000 training set'
                       'examples will be set aside for validation.',
                  dest='validation',
                  action='store_true',
                  default=False)
argp.add_argument('--no-validation',
                  help='Flag - use to stop validation being run.',
                  dest='validation',
                  action='store_false')
argp.add_argument('--testing',
                  help='Flag - whether to run model evaluation on the test set.',
                  dest='testing',
                  action='store_true',
                  default=True)
argp.add_argument('--no-testing',
                  help='Flag - use to stop testing being run.',
                  dest='testing',
                  action='store_false')
argp.add_argument('-l', '--learning-rate',
                  help='Learning rate at start of training. This will override the config file.'
                       'The learning rate applied to the std network of the encoder will be half'
                       'that of the other networks (and half of what is specified here).',
                  type=float,
                  default=None)
argp.add_argument('-c', '--config-file',
                  help='Path to the config file to be used for this run.'
                       'Set to either A. "configs/default_config.json" to use set up as in paper, '
                       'B. "configs/global_cv_config.json" is as in paper but with scalar control variate, '
                       'C. "configs/bs_experiment_config.json" to use hyperparameters as in batch size experiments, '
                       'D. "<path/to/custom_config>.json" if you want to try a custom arrangement.',
                  type=str,
                  default='configs/default_config.json')
argp.add_argument("--log-level", type=str, default='INFO',
                  choices=['NOTSET', 'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'],
                  help="Set the logging level")
argp.add_argument('--only-ipu',
                  help='restrict execution target to IPU only',
                  action='store_true')
args = argp.parse_args()

# Model parameters
TUNED_LEARNING_RATE = {16: 5e-4,    # Batch size -> Learning rate
                       100: 5e-4,
                       128: 9e-4,
                       512: 9e-4,
                       1024: 13e-4}
EPOCH_TIMESCALE_LR = 30
EPOCHS_ORIGINAL = 800
EXAMPLES_PER_EPOCH = 50000

VALIDATION_SIZE = 10000   # Only used if args.validation is True


def update_model_config(conf):
    """Update the model configuration with the parameters specified above"""
    conf[0]['args']['micro_batch_size'] = args.micro_batch_size

    # Calculate number of iterations from number of epochs and batch size
    conf[0]['args']['training']['n_iter'] = \
        int(conf[0]['args']['training']['n_epoch'] * EXAMPLES_PER_EPOCH / args.micro_batch_size)

    # How often to do validation and testing
    conf[0]['args']['training']['n_batch_freq_val'] = \
        int(conf[0]['args']['training']['n_epoch_freq_val'] * EXAMPLES_PER_EPOCH / args.micro_batch_size)
    conf[0]['args']['testing']['n_batch_freq_test'] = \
        int(conf[0]['args']['testing']['n_epoch_freq_test'] * EXAMPLES_PER_EPOCH / args.micro_batch_size)

    if args.learning_rate is None:
        if conf[0]['args']['training']['base_lr']['encoder']['mean'] == "":
            conf[0]['args']['training']['base_lr']['encoder']['mean'] = TUNED_LEARNING_RATE[args.micro_batch_size]
        if conf[0]['args']['training']['base_lr']['encoder']['std'] == "":
            conf[0]['args']['training']['base_lr']['encoder']['std'] = TUNED_LEARNING_RATE[args.micro_batch_size] * 0.5
        if conf[0]['args']['training']['base_lr']['decoder'] == "":
            conf[0]['args']['training']['base_lr']['decoder'] = TUNED_LEARNING_RATE[args.micro_batch_size]
    else:
        conf[0]['args']['training']['base_lr'] = {'encoder':
                                                  {'mean': args.learning_rate,
                                                   'std': args.learning_rate * 0.5},
                                                  'decoder': args.learning_rate}

    # Rescale HMC step size adaption rate proportional to batch size
    conf[0]['args']['training']['mcmc']['step_size_adaption_rate'] = \
        float(args.micro_batch_size) / 100. * conf[0]['args']['training']['mcmc']['step_size_adaption_rate']
    conf[0]['args']['results_location'] = args.results_dir
    conf[0]['args']['only_ipu'] = args.only_ipu if args.only_ipu else False
    conf[0]['args']['task_name'] = args.run_name
    conf[0]['args']['data']['n_validation'] = VALIDATION_SIZE if args.validation else 0

    # Log-level from command line
    conf[0]['args']['log'] = {'level': args.log_level}
    return conf


def update_experiment_config(conf):
    """Update experiment config with storage locations"""
    conf['args']['results_location'] = args.results_dir
    run_name = args.run_name or os.path.basename(args.config_file).replace('.json', '')
    subdir = run_name
    rep = 0
    while os.path.exists(os.path.join(args.results_dir, subdir)):
        subdir = f'{run_name}_{rep}'
        rep += 1
    conf['args']['task_name'] = subdir
    conf['args']['validation'] = args.validation
    conf['args']['testing'] = args.testing
    conf['args']['data']['n_validation'] = VALIDATION_SIZE if args.validation else 0

    # Log-level from command line
    conf['args']['log'] = {'level': args.log_level}
    return conf


if __name__ == '__main__':
    for _ in range(args.repeats):
        with open(args.config_file) as f:
            config = json.load(f)

        exp_config = config['experiment']
        model_config = config['model']
        model_config = update_model_config(model_config)
        exp_config = update_experiment_config(exp_config)

        g = Experiment(exp_config)
        g.execute(model_config)
