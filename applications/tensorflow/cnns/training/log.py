# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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
The logging code used in train.py.
"""

import os
import csv
import random
import string
import json
import wandb
import tensorflow as tf

try:
    # See
    # https://github.com/mlcommons/logging/blob/1.0.0/mlperf_logging/mllog/examples/dummy_example.py
    # for further details on mlperf logging configurations
    from mlperf_logging import mllog

    MLPERF_LOGGING = True
    MLLOGGER = mllog.get_mllogger()
except ImportError:
    MLPERF_LOGGING = False


def add_arguments(parser):
    group = parser.add_argument_group('Logging')
    group.add_argument('--log-dir', type=str, default="./logs/",
                       help="Log and weights save directory")
    group.add_argument('--logs-path', type=str, default=None,
                       help="Log and weights save directory for current run.")
    group.add_argument('--name-suffix', type=str,
                       help="Suffix added to name string")
    group.add_argument('--logs-per-epoch', type=int, default=16,
                       help="Logs per epoch (if number of epochs specified)")
    group.add_argument('--log-freq', type=int, default=500,
                       help="Log statistics every N mini-batches (if number of iteration specified)")
    group.add_argument('--no-logs', action='store_true',
                       help="Don't create any logs")
    group.add_argument('--log-all-instances', type=bool,
                       help="""Allow all instances to log results.
                             By default only instance 0 creates logs.""")
    group.add_argument('--mlperf-logging', action='store_true',
                       help="Activate MLPerf logging if installed.")
    group.add_argument("--wandb", action="store_true",
                       help="Enable logging to Weights and Biases.")
    group.add_argument("--wandb-project", type=str, default="tf-cnn",
                       help="Configures project Weights and Biases logs to.")
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

    # only instance 0 creates a log dir and logs to disk
    # a log dir is also created when using validation.py (aka opts['training']==False)
    # using train.py with --restore-path logs training results into that folder
    if ((not opts['no_logs']) and
            (not opts['restore_path'] or not opts.get('training')) and
            (opts['distributed_worker_index'] == 0 or opts['log_all_instances'])):
        if "logs_path" not in opts or opts["logs_path"] is None:
            opts["logs_path"] = os.path.join(opts["log_dir"], '{}'.format(name))

        opts["checkpoint_path"] = os.path.join(opts["logs_path"], "ckpt")

        if not os.path.isdir(opts["logs_path"]):
            os.makedirs(opts["logs_path"])

        opts['summary_str'] += " Saving to {logs_path}\n"

        fname = os.path.join(opts["logs_path"], 'arguments.json')
        if os.path.isfile(fname):
            fname = os.path.join(opts["logs_path"], 'arguments_restore.json')
        with open(fname, 'w') as fp:
            json.dump(opts, fp, sort_keys=True, indent=4, separators=(',', ': '))
    elif (opts['restore_path'] and
            (opts['distributed_worker_index'] == 0 or opts['log_all_instances'])):
        opts['logs_path'] = opts['restore_path']
        opts['checkpoint_path'] = os.path.join(opts['logs_path'], 'ckpt')
    else:
        opts["logs_path"] = None
        opts["log_dir"] = None
        opts["mlperf_logging"] = False
        opts["checkpoint_path"] = os.path.join('/tmp/', '{}/ckpt'.format(name))
        if not os.path.isdir(os.path.dirname(os.path.abspath(opts["checkpoint_path"]))):
            os.makedirs(os.path.dirname(os.path.abspath(opts["checkpoint_path"])))

    global MLPERF_LOGGING
    if opts["mlperf_logging"] and MLPERF_LOGGING and opts['distributed_worker_index'] == 0:
        MLPERF_LOGGING = True
        seed = opts.get("seed", "None")
        try:
            mllog.config(
                default_namespace=mllog.constants.RESNET,
                default_stack_offset=2,
                default_clear_line=False,
                root_dir=os.path.split(os.path.abspath(__file__))[0],
                filename=os.path.join(opts["logs_path"],
                                      "result_{}.txt".format(seed))
            )
        except NameError:
            pass
    else:
        MLPERF_LOGGING = False

    return opts


def print_to_file_and_screen(string, opts):
    print(string)
    if (opts["logs_path"] and
            (opts['distributed_worker_index'] == 0 or opts['log_all_instances'])):
        with open(os.path.join(opts["logs_path"], 'log.txt'), "a+") as f:
            f.write(str(string) + '\n')


def write_to_csv(d, write_header, training, opts):
    if (opts["logs_path"] and
            (opts['distributed_worker_index'] == 0 or opts['log_all_instances'])):
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


def mlperf_logging(key, value=None, log_type="event", metadata=None):
    if not MLPERF_LOGGING:
        return
    if key not in mllog.constants.__dict__:
        import warnings
        warnings.warn("Provided MLPERF key ('{}') is not supported.".format(key))
        import time
        time.sleep(10)
        key = key.lower()
    else:
        key = mllog.constants.__dict__[key]
    if log_type is "start":
        MLLOGGER.start(
            key=key, value=value, metadata=metadata)
    elif log_type is "event":
        MLLOGGER.event(
            key=key, value=value, metadata=metadata)
    elif log_type is "stop" or log_type is "end":
        MLLOGGER.end(
            key=key, value=value, metadata=metadata)
    else:
        raise NotImplementedError("Unknown log type {}".format(log_type))


def initialise_wandb(opts):
    """Initialises weights and biases run with model options"""
    project = opts["wandb_project"]
    name = opts["name"]
    name_suffix = opts.get("name_suffix", None)
    if name_suffix:
        name += name_suffix
    wandb.init(project=project, name=name, sync_tensorboard=True)
    wandb.config.update(opts)


def log_to_wandb(stats):
    """Logs stats to weights and biases run"""
    wandb.log(stats)
