# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Logging utilities.
"""

import csv
import datetime
import json
import logging
import os
import random
import subprocess

import numpy as np
import tensorflow as tf
from tensorflow import pywrap_tensorflow

# Set Python logger
# Match TensorFlow's default logging format.
logFormatter = logging.Formatter(
    '%(asctime)s.%(msecs)06d: %(levelname)-1.1s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


def get_logger():
    return logger


def set_log_file_path(log_file_path):
    global logger
    fileHandler = logging.FileHandler(log_file_path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)


def add_arguments(parser):
    group = parser.add_argument_group('Logging')
    group.add_argument('--log-dir', type=str, default="./logs/",
                       help="Log and weights save directory")
    group.add_argument('--name-suffix', type=str,
                       help="Suffix added to name string")
    group.add_argument('--steps-per-logs', type=int, default=1,
                       help="Logs per epoch (if number of epochs specified)")
    group.add_argument('--steps-per-tensorboard', type=int, default=0,
                       help='Number of steps between saving statistics to TensorBoard. 0 to disable.')
    return parser


def set_defaults(opts):
    name = opts['name']

    if opts["name_suffix"]:
        name = name + "_" + opts["name_suffix"]

    if opts.get("poplar_version"):
        v = opts['poplar_version']
        # name += "_v" + v[v.find("version ") + 8: v.rfind(' ')]
        name += "_v" + v[v.find("version ") + 8: v.find(' (')]

    # We want this to be random even if random seeds have been set so that we don't overwrite
    # when re-running with the same seed
    random_state = random.getstate()
    random.seed()
    random.setstate(random_state)

    # System time with milliseconds
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    name += "_{}".format(time)

    if not os.path.isdir(opts["save_path"]):
        os.makedirs(opts["save_path"], exist_ok=True)

    opts["logs_path"] = os.path.join(opts["save_path"], name)
    opts["checkpoint_path"] = os.path.join(opts["save_path"], name, 'ckpt')

    if not os.path.isdir(opts["logs_path"]):
        os.makedirs(opts["logs_path"], exist_ok=True)

    set_log_file_path(os.path.join(opts['logs_path'], 'log.txt'))

    with open(os.path.join(opts["logs_path"], 'arguments.json'), 'w') as fp:
        json.dump(opts, fp, sort_keys=True, indent=4, separators=(',', ': '))
    return opts


def write_to_csv(d, write_header, training, logs_path):
    if logs_path:
        filename = 'training.csv' if training else 'validation.csv'
        with open(os.path.join(logs_path, filename), 'a+') as f:
            w = csv.DictWriter(f, d.keys())
            if write_header:
                w.writeheader()
            w.writerow(d)


def print_trainable_variables(logs_path):
    logger.info('Trainable Variables:')
    total_parameters = 0
    for variable in tf.trainable_variables():
        logger.info(variable)
        variable_parameters = 1
        for DIM in variable.get_shape():
            variable_parameters *= DIM.value
        total_parameters += variable_parameters
    logger.info('Total Parameters:' + str(total_parameters) + '\n')


def make_histogram(values, bins=512):
    # From https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    # License: BSD License 2.0
    # Author Michael Gygli

    # Logs the histogram of a list/vector of values.
    # Convert to a numpy array
    values = np.array(values)

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    # Create and write Summary
    return hist
    # return tf.Summary.Value(tag=tag, histo=hist)


def save_model_statistics(checkpoint_path, summary_writer, step=0):
    initializers = load_initializers_from_checkpoint(checkpoint_path)
    summary = tf.Summary()
    for name, np_weight in initializers.items():
        name = name.replace(":", "_")
        tensor = np_weight.astype(np.float32)
        if not np.any(np.isnan(tensor)):
            summary.value.add(tag=name, histo=make_histogram(tensor))
            summary.value.add(tag=f"L2/{name}", simple_value=np.linalg.norm(tensor))

    summary_writer.add_summary(summary, step)
    summary_writer.flush()


def load_initializers_from_checkpoint(checkpoint_path):
    initializers = {}
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_map = reader.get_variable_to_shape_map()
    for key, dim in var_to_map.items():
        if key == 'global_step':
            continue
        # if reader.get_tensor(key).dtype.name == 'float16':
        #     int_data = np.asarray(reader.get_tensor(key), np.int32)
        #     np_weight = int_data.view(dtype=np.float16).reshape(dim)
        # else:
        np_weight = reader.get_tensor(key)
        initializers[key] = np_weight
    return initializers


def get_git_revision():
    return subprocess.check_output(["git", "describe", "--always", "--dirty"]).strip().decode()
