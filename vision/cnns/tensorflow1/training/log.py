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
from typing import NamedTuple
import os
import pathlib
import csv
import random
import string
import json
import wandb
import tensorflow as tf
import logging

import numpy as np

import pva

try:
    # See
    # https://github.com/mlcommons/logging/blob/1.0.0/mlperf_logging/mllog/examples/dummy_example.py
    # for further details on mlperf logging configurations
    from mlperf_logging import mllog

    MLPERF_LOGGING = True
    MLLOGGER = mllog.get_mllogger()
except ImportError:
    MLPERF_LOGGING = False

GCL_METHODS = {"auto", "broadcast", "clockwise_ring", "anticlockwise_ring", "bidirectional_ring_pair", "meet_in_middle_ring", "quad_directional_ring"}


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
    group.add_argument("--profile", action="store_true",
                       help="Turns on graph profiling and saves memory profiles to wandb if it is active."
                            " This option will stop execution after a single device iteration.")
    group.add_argument("--profile-compilation", action="store_true",
                       help="Turns on graph profiling for the compilation, "
                            "saves memory profiles to wandb if it is active.")
    group.add_argument("--gcl-method", type=str,
                       help=f'Set the method in the GCL_OPTIONS environment variable. Must be one of: {GCL_METHODS}')
    group.add_argument("--gcl-max-broadcast-size", type=int,
                       help='Set maxBroadcastSize in the GCL_OPTIONS environment variable.')
    group.add_argument("--executable-cache-path", type=str, default="",
                       help="Set the executable cache path for the poplar_executable")
    group.add_argument("--poplar-sync-configuration", type=str, default="",
                       choices={"", "ipuAndInstanceAndIntraReplicaAndAll"},
                       help=r"Sets the POPLAR_TARGET_OPTIONS='{syncConfiguration: ''} environment variable")
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


def mlperf_logging(key, value=None, log_type="event", metadata=None, stack_offset=None):
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
    if log_type == "start":
        MLLOGGER.start(
            key=key, value=value, metadata=metadata, stack_offset=stack_offset)
    elif log_type == "event":
        MLLOGGER.event(
            key=key, value=value, metadata=metadata, stack_offset=stack_offset)
    elif log_type == "stop" or log_type == "end":
        MLLOGGER.end(
            key=key, value=value, metadata=metadata, stack_offset=stack_offset)
    else:
        raise NotImplementedError("Unknown log type {}".format(log_type))


def initialise_wandb(opts):
    """Initialises weights and biases run with model options"""
    project = opts["wandb_project"]
    name = opts["name"]
    name_suffix = opts.get("name_suffix", None)
    if name_suffix:
        name += name_suffix

    wandb_id = opts.get("wandb_id", None)
    if wandb_id is None:
        wandb.init(project=project, name=name, sync_tensorboard=True)
        wandb.config.update(opts)
    else:
        wandb.init(id=wandb_id, project=project, resume="must", sync_tensorboard=True)
        wandb.run.summary["restored_config"] = opts


def log_to_wandb(stats, commit=None):
    """Logs stats to weights and biases run"""
    wandb.log(stats, commit=commit)


def add_to_wandb_summary(opts, column, value):
    """Add a column to the wandb summary"""
    if opts['wandb'] and opts['distributed_worker_index'] == 0:
        wandb.run.summary[column] = value


def handle_profiling_options(opts):
    """Processes the `--profile` and `--profile-compilation` options

    Turns on profiling and updates the POPLAR_ENGINE_OPTIONS environment.
    """
    logging_dir = opts["logs_path"] if opts["logs_path"] else "./profile"

    if opts["profile"] and opts["profile_compilation"]:
        raise ValueError("Both `--profile` and `--profile-compilation` options were "
                         "used but they are not compatible.")

    if opts["profile"] or opts["profile_compilation"]:
        engine_options = {
                "debug.allowOutOfMemory": "true",
                "autoReport.directory": logging_dir,
                "autoReport.all": "true",
        }
        if opts["profile_compilation"]:
            engine_options["autoReport.outputExecutionProfile"] = "false"

        # Update the configuration set by the POPLAR_ENGINE_OPTIONS environment variable
        engine_options.update(json.loads(os.environ.get("POPLAR_ENGINE_OPTIONS", "{}")))
        logging_dir = engine_options["autoReport.directory"]
        logging.info(f"Profile files will be available in {logging_dir}. Set --logs_path"
                     " or the POPLAR_ENGINE_OPTIONS environment variable to change this.")
        engine_options_str = json.dumps(engine_options)
        logging.info(f"    Engine options: {engine_options_str}.")
        os.environ["POPLAR_ENGINE_OPTIONS"] = engine_options_str


def handle_gcl_options(opts):
    """
    Processes the `--gcl-method` and `--gcl-max-broadcast-size` option

    Updates the GCP_OPTIONS environment.
    """
    if opts.get("gcl_method") or opts.get("gcl_max_broadcast_size"):
        gcl_options = {}
        if opts["gcl_method"]:
            if opts["gcl_method"] not in GCL_METHODS:
                raise ValueError(f"The argument of --gcl-broadcast must be one of {GCL_METHODS}")
            gcl_options["method"] = opts["gcl_method"]

        if opts["gcl_max_broadcast_size"]:
            gcl_options["syncful.maxBroadcastSize"] = opts["gcl_max_broadcast_size"]

        # Update the configuration set by the GCL_OPTIONS environment variable
        gcl_options.update(json.loads(os.environ.get("GCL_OPTIONS", "{}")))
        gcl_options_str = json.dumps(gcl_options)
        logging.info(f"    GCL options: {gcl_options_str}.")
        os.environ["GCL_OPTIONS"] = gcl_options_str


def handle_cache_path(opts):
    if opts.get("executable_cache_path"):
        tf_poplar_flags = os.environ.get("TF_POPLAR_FLAGS", "")
        if "executable_cache_path" in tf_poplar_flags:
            raise ValueError("The executable cache path was set as an environemnt variable in TF_POPLAR_FLAGS"
                             f" ({tf_poplar_flags}) and as an argument.")
        os.makedirs(opts["executable_cache_path"], exist_ok=True)
        new_cache_path = opts["executable_cache_path"]
        os.environ["TF_POPLAR_FLAGS"] = f"{tf_poplar_flags} --executable_cache_path={new_cache_path}"


def handle_poplar_target_options(opts):
    """
    Processes the `--poplar-sync-configuration` option

    Updates the POPLAR_TARGET_OPTIONS environment.
    """

    if opts.get("poplar_sync_configuration"):
        poplar_target_options = {}
        if opts.get("poplar_sync_configuration"):
            poplar_target_options["syncConfiguration"] = opts["poplar_sync_configuration"]

        # Update the configuration set by the poplar_target_options environment variable
        poplar_target_options.update(json.loads(os.environ.get("POPLAR_TARGET_OPTIONS", "{}")))
        poplar_target_options_str = json.dumps(poplar_target_options)
        logging.info(f"    Poplar target options: {poplar_target_options}.")
        os.environ["POPLAR_TARGET_OPTIONS"] = poplar_target_options_str



class SummarisedArray(NamedTuple):
    """Calculate and store summary metrics for an array"""
    min: float
    max: float
    mean: float

    @classmethod
    def from_array(cls, array: np.ndarray):
        array = np.array(array)
        return cls(
            min=array.min(),
            max=array.max(),
            mean=array.mean(),
        )


def process_profile(opts):
    """Logs the IPU memory profile to wandb"""
    logging_dir = opts["logs_path"] if opts["logs_path"] else "./profile"

    profile_paths = [*pathlib.Path(logging_dir).rglob("*.pop")]
    if len(profile_paths) > 1:
        logging.warning("Multiple Graph Analyzer profiles detected, only the first "
                        f"will be processed: {profile_paths[0]}")
    elif not profile_paths:
        if opts["profile"]:
            logging.warn(f"No profile found in {logging_dir} despite `--profile`.")
        return

    report = pva.openReport(str(profile_paths[0]))

    always_live_memory = sum(v.size for v in report.compilation.alwaysLiveVariables)
    not_always_live = [
        step.notAlwaysLiveMemory.bytes
        for step in report.compilation.livenessProgramSteps
    ]
    total_live_memory = np.array(always_live_memory) + not_always_live
    tile_memory = [tile.memory.total.includingGaps for tile in report.compilation.tiles]
    single_tile_mem = 624 * 1024
    profile_data = dict(
        not_always_live=np.array(not_always_live),
        total_live_memory = total_live_memory,
        free_live_memory = (single_tile_mem * 1472) - total_live_memory,
        tile_memory=np.array(tile_memory),
        tile_free_memory=single_tile_mem - np.array(tile_memory),
    )

    if not opts["wandb"] or opts['distributed_worker_index'] != 0:
        return

    data = [[x, y, z] for (x, (y, z)) in enumerate(zip(not_always_live, total_live_memory))]
    table = wandb.Table(data=data, columns = ["program_step", "not_always_live_memory", "total_live_memory"])
    wandb.log({"memory_liveness": wandb.plot.line(table, "program_step", "not_always_live_memory",
               title="Memory Liveness")})
    wandb.log({"memory_total_liveness": wandb.plot.line(table, "program_step", "total_live_memory",
               title="Total Memory Liveness")})

    data = [[x, y] for (x, y) in enumerate(tile_memory)]
    table = wandb.Table(data=data, columns = ["tile", "tile_memory_usage"])
    wandb.log({"tile_memory": wandb.plot.line(table, "tile", "tile_memory_usage",
               title="Tile memory")})

    for k, v in profile_data.items():
        wandb.run.summary[k] = v
    # Write a custom summary, makes sure that these metrics are available for reporting in wandb
    wandb.run.summary["summaries"] = {
        k: SummarisedArray.from_array(v)._asdict()
        for k, v in profile_data.items()
    }
