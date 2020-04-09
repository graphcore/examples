# Copyright 2019 Graphcore Ltd.
"""
The logging code used in train.py.
"""

import os
import csv
import random
import string
import json
import tensorflow as tf


def add_arguments(parser):
    group = parser.add_argument_group('Logging')
    group.add_argument('--log-dir', type=str, default="./logs/",
                       help="Log and weights save directory")
    group.add_argument('--name-suffix', type=str,
                       help="Suffix added to name string")
    group.add_argument('--logs-per-epoch', type=int, default=16,
                       help="Logs per epoch (if number of epochs specified)")
    group.add_argument('--log-freq', type=int, default=500,
                       help="Log statistics every N mini-batches (if number of iteration specified)")
    group.add_argument('--no-logs', action='store_true',
                       help="Don't create any logs")
    return parser


def _extract_poplar_version(v):
    prefix = "version "
    start = v.index(prefix) + len(prefix)
    end = v.index(" ", start)
    return v[start:end]


def set_defaults(opts):
    # Logs and checkpoint paths
    # Must be run last
    opts['summary_str'] += "Logging\n"
    name = opts['name']

    if opts["name_suffix"]:
        name = name + "_" + opts["name_suffix"]

    if opts.get("poplar_version"):
        name += "_v" + _extract_poplar_version(opts['poplar_version'])

    # We want this to be random even if random seeds have been set so that we don't overwrite
    # when re-running with the same seed
    random_state = random.getstate()
    random.seed()
    rnd_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(3))
    random.setstate(random_state)
    name += "_{}".format(rnd_str)
    opts['summary_str'] += " Name: {name}\n"

    if not opts['no_logs']:
        opts["logs_path"] = os.path.join(opts["log_dir"], '{}'.format(name))
        opts["checkpoint_path"] = os.path.join(opts["log_dir"], '{}/ckpt'.format(name))

        if not os.path.isdir(opts["logs_path"]):
            os.makedirs(opts["logs_path"])

        opts['summary_str'] += " Saving to {logs_path}\n"
        with open(os.path.join(opts["logs_path"], 'arguments.json'), 'w') as fp:
            json.dump(opts, fp, sort_keys=True, indent=4, separators=(',', ': '))
    else:
        opts["logs_path"] = None
        opts["log_dir"] = None
        opts["checkpoint_path"] = os.path.join('/tmp/', '{}/ckpt'.format(name))
        if not os.path.isdir(os.path.dirname(os.path.abspath(opts["checkpoint_path"]))):
            os.makedirs(os.path.dirname(os.path.abspath(opts["checkpoint_path"])))

    return opts


def print_to_file_and_screen(string, opts):
    print(string)
    if opts["logs_path"]:
        with open(os.path.join(opts["logs_path"], 'log.txt'), "a+") as f:
            f.write(str(string) + '\n')


def write_to_csv(d, write_header, training, opts):
    if opts["logs_path"]:
        filename = 'training.csv' if training else 'validation.csv'
        with open(os.path.join(opts['logs_path'], filename), 'a+') as f:
            w = csv.DictWriter(f, d.keys())
            if write_header:
                w.writeheader()
            w.writerow(d)


def print_trainable_variables(opts):
    print_to_file_and_screen('Trainable Variables:', opts)
    total_parameters = 0
    for variable in tf.trainable_variables():
        print_to_file_and_screen(variable, opts)
        variable_parameters = 1
        for DIM in variable.get_shape():
            variable_parameters *= DIM.value
        total_parameters += variable_parameters
    print_to_file_and_screen('Total Parameters:' + str(total_parameters) + '\n', opts)
