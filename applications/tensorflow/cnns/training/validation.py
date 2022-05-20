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
The validation code used in train.py.

This script can also be called to run validation on previously generated checkpoints.
See the README for more information.

"""

import tensorflow as tf
import os
import re
import time
import json
import argparse
import sys
import csv
from collections import OrderedDict
import importlib
from glob import glob
from pathlib import Path
import threading

import train
import log as logging
from Datasets import data as dataset
from Datasets import imagenet_dataset
from tensorflow.python import ipu
from ipu_utils import get_config
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import loops, ipu_infeed_queue, ipu_outfeed_queue
import tensorflow.contrib.compiler.xla as xla
from tensorflow.python.ipu.ops import cross_replica_ops
from tensorflow.python.ipu import horovod as hvd
import popdist
import popdist.tensorflow
import configurations
import relative_timer

DATASET_CONSTANTS = dataset.DATASET_CONSTANTS
MLPERF_EVAL_TARGET = 75.9


class LatencyThread:

    def __init__(self, valid, total_batches):
        self.thread = None
        self.valid = valid
        self.total_batches = total_batches
        self.latency_sum = 0.
        self.start = self.__fake_start
        self.__start_if_not_compiled = self.__real_start

    def __fake_start(self):
        pass

    def __real_start(self):
        self.thread = threading.Thread(target=self.compute_latency)
        self.thread.start()

    def __setup_real_start(self):
        self.start = self.__real_start
        self.__start_if_not_compiled = self.__fake_start

    def join(self):
        # call to first join indicates compilation complete
        # call start thread and setup real start
        self.__start_if_not_compiled()
        self.__setup_real_start()

        self.thread.join()

    def compute_latency(self):
        num_batches = 0
        self.latency_sum = 0
        while num_batches < self.total_batches:
            latencies = self.valid.session.run(self.valid.ops['latency_per_batch'])
            num_batches += latencies.shape[0]
            self.latency_sum += latencies.sum()

    def get_latency(self):
        return self.latency_sum / self.total_batches if self.total_batches != 0 else -0.001


def validation_graph_builder(model, data_dict, opts):
    train.ipuside_preprocessing(data_dict, opts, training=False)
    image, label = data_dict['image'], data_dict['label']
    logits = model(opts, training=False, image=image)
    predictions = tf.argmax(logits, 1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, label), tf.float16))
    return accuracy


def validation_graph(model, opts):
    reconfigure = not opts.get('reuse_IPUs', False)
    if opts['use_popdist'] and reconfigure:
        hvd.init()

    valid_graph = tf.Graph()
    with valid_graph.as_default():
        # datasets must be defined outside the ipu device scope
        valid_dataset = dataset.data(opts, is_training=False).map(lambda x: {'data_dict': x})

        valid_iterator = ipu_infeed_queue.IPUInfeedQueue(valid_dataset,
                                                         prefetch_depth=opts['prefetch_depth'])

        if opts['latency']:
            timestamp_queue = ipu_outfeed_queue.IPUOutfeedQueue()

        with ipu_scope('/device:IPU:0'):
            def comp_fn():
                def body(total_accuracy, data_dict):
                    accuracy = validation_graph_builder(model, data_dict, opts)
                    if opts['latency']:
                        timestamp_enqueue = timestamp_queue.enqueue(data_dict['timestamp'])
                        return (total_accuracy + (tf.cast(accuracy, tf.float32) / opts["validation_batches_per_step"]),
                                timestamp_enqueue)
                    else:
                        return total_accuracy + (tf.cast(accuracy, tf.float32) / opts["validation_batches_per_step"])
                accuracy = loops.repeat(int(opts["validation_batches_per_step"]),
                                        body, [tf.constant(0, tf.float32)], valid_iterator)
                if opts['total_replicas'] * opts['shards'] > 1 and not opts.get('inference', False):
                    accuracy = cross_replica_ops.cross_replica_sum(accuracy) / (opts['total_replicas'] * opts['shards'])
                return accuracy

            (accuracy,) = xla.compile(comp_fn, [])

        accuracy = 100 * accuracy

        if opts['latency']:
            print(f'relative_timer start {relative_timer.get_start()}')
            timestamp = tf.cast(tf.timestamp() - relative_timer.get_start(), tf.float32)
            latency_per_batch = tf.reshape(timestamp - timestamp_queue.dequeue(), [-1])
        else:
            latency_per_batch = None

        valid_saver = tf.train.Saver()

        ipu.utils.move_variable_initialization_to_cpu()
        valid_init = tf.global_variables_initializer()

        if opts['use_popdist']:
            broadcast_weights = []
            for var in tf.global_variables():
                broadcast_weights.append(var.assign(hvd.broadcast(var, root_rank=0)))
            global_batch_size_ph = tf.placeholder(dtype=tf.int32, shape=())
            broadcast_global_batch_size = hvd.broadcast(global_batch_size_ph, root_rank=0)
            num_files_ph = tf.placeholder(dtype=tf.int32, shape=())
            broadcast_num_files = hvd.broadcast(num_files_ph, root_rank=0)
            iteration_ph = tf.placeholder(dtype=tf.int32, shape=())
            broadcast_iteration = hvd.broadcast(iteration_ph, root_rank=0)
        else:
            broadcast_weights = None
            broadcast_global_batch_size, global_batch_size_ph = None, None
            broadcast_num_files, num_files_ph = None, None
            broadcast_iteration, iteration_ph = None, None

    globalAMP = None
    if opts["available_memory_proportion"] and len(opts["available_memory_proportion"]) == 1:
        globalAMP = opts["available_memory_proportion"][0]

    ipu_options = get_config(ipu_id=opts["select_ipu"],
                             stochastic_rounding='OFF',  # disable Stochastic Rounding for validation
                             shards=opts['shards'],
                             number_of_replicas=opts['total_replicas'],
                             max_cross_replica_buffer_size=opts["max_cross_replica_buffer_size"],
                             fp_exceptions=opts["fp_exceptions"],
                             half_partials=opts["enable_half_partials"],
                             conv_dithering=opts["enable_conv_dithering"],
                             enable_recomputation=opts["enable_recomputation"],
                             seed=opts["seed"],
                             availableMemoryProportion=globalAMP,
                             stable_norm=opts["stable_norm"],
                             compile_only=opts["compile_only"],
                             internalExchangeOptimisationTarget=opts[
                                 "internal_exchange_optimisation_target"
                             ],
                             num_io_tiles=opts["num_io_tiles"],
                             number_of_distributed_batch_norm_replicas=opts.get("BN_span", 1),
                             nanoo=not opts["saturate_on_overflow"],
                             )

    if opts['use_popdist'] and reconfigure:
        ipu_options = popdist.tensorflow.set_ipu_config(ipu_options, opts['shards'], configure_device=False)

    if opts['on_demand'] and reconfigure:
        ipu_options.device_connection.enable_remote_buffers = True
        ipu_options.device_connection.type = ipu.utils.DeviceConnectionType.ON_DEMAND

    if reconfigure:
        ipu_options.configure_ipu_system()

    valid_sess = tf.Session(graph=valid_graph, config=tf.ConfigProto())

    ops = {'accuracy': accuracy,
           'broadcast_weights': broadcast_weights,
           'broadcast_global_batch_size': broadcast_global_batch_size,
           'broadcast_num_files': broadcast_num_files,
           'broadcast_iteration': broadcast_iteration,
           'latency_per_batch': latency_per_batch}

    placeholders = {'global_batch_size': global_batch_size_ph,
                    'num_files': num_files_ph,
                    'iteration': iteration_ph}

    valid_graph.finalize()

    return train.GraphOps(valid_graph, valid_sess, valid_init, ops, placeholders, valid_iterator, None, valid_saver)


def validation_run(valid, filepath, i, epoch, first_run, opts, latency_thread):
    run = True
    if filepath:
        valid.saver.restore(valid.session, filepath)
        name = filepath.split('/')[-1]

        csv_path = os.path.join(opts['logs_path'], 'validation.csv')
        if os.path.exists(csv_path):
            with open(csv_path, 'rU') as infile:
                # read the file as a dictionary for each row ({header : value})
                reader = csv.DictReader(infile)
                for row in reader:
                    if row['name'] == name:
                        run = False
                        print('Skipping validation run on checkpoint: {}'.format(name))
                        break
    else:
        name = None

    if run:
        if opts['use_popdist']:
            # synchronise the model weights across all instances
            valid.session.run(valid.ops['broadcast_weights'])

        logging.mlperf_logging(key="EVAL_START", log_type="start",
                               metadata={"epoch_num": round(epoch)})
        # Gather accuracy statistics
        accuracy = 0.0

        # start latency thread
        latency_thread.start()

        start = relative_timer.now()
        for __ in range(opts["validation_iterations"]):
            try:
                a = valid.session.run(valid.ops['accuracy'])
            except tf.errors.OpError as e:
                if opts['compile_only'] and 'compilation only' in e.message:
                    print("Validation graph successfully compiled")
                    print("Exiting...")
                    sys.exit(0)
                raise tf.errors.ResourceExhaustedError(e.node_def, e.op, e.message)

            accuracy += a
        val_time = relative_timer.now() - start
        accuracy /= opts["validation_iterations"]

        # wait for all dequeues and latency computation
        latency_thread.join()
        latency = latency_thread.get_latency()

        valid_format = (
            "Validation top-1 accuracy [{name}] (iteration: {iteration:6d}, epoch: {epoch:6.2f}, img/sec: {img_per_sec:6.2f},"
            " time: {val_time:8.6f}, latency (ms): {latency:8.4f}): {val_acc:6.3f}%")

        val_size = (opts["validation_iterations"] *
                    opts["validation_batches_per_step"] *
                    opts["validation_global_batch_size"])

        count = int(DATASET_CONSTANTS[opts['dataset']]['NUM_VALIDATION_IMAGES'])

        raw_accuracy = accuracy
        if count < val_size:
            accuracy = accuracy * val_size / count

        stats = OrderedDict([
                    ('name', name),
                    ('iteration', i),
                    ('epoch', epoch),
                    ('val_acc', accuracy),
                    ('raw_acc', raw_accuracy),
                    ('val_time', val_time),
                    ('val_size', val_size),
                    ('img_per_sec', val_size / val_time),
                    ('latency', latency * 1000),
                ])
        logging.print_to_file_and_screen(valid_format.format(**stats), opts)
        logging.write_to_csv(stats, first_run, False, opts)
        if opts["wandb"] and opts["distributed_worker_index"] == 0:
            logging.log_to_wandb(stats)
        logging.mlperf_logging(key="EVAL_STOP", log_type="stop",
                               metadata={"epoch_num": round(epoch)})
        logging.mlperf_logging(
            key="EVAL_ACCURACY", value=float(stats["val_acc"]) / 100,
            metadata={"epoch_num": round(epoch)})
        return stats


def initialise_validation(model, opts):
    # -------------- BUILD GRAPH ------------------
    valid = validation_graph(model.Model, opts)
    # ------------- INITIALIZE SESSION -----------

    valid.session.run(valid.iterator.initializer)
    with valid.graph.as_default():
        valid.session.run(valid.init)

    return valid


def validation_only_process(model, opts):
    valid = initialise_validation(model, opts)

    ckpt_pattern_idx = re.compile(".*ckpt-([0-9]+).index$")
    ckpt_pattern = re.compile(".*ckpt-([0-9]+)$")
    if opts["restore_path"] and opts['distributed_worker_index'] == 0:
        if os.path.isdir(opts["restore_path"]):
            # search to a maximum depth of 1
            ckpts = glob(os.path.join(opts["restore_path"], '*.index')) \
                + glob(os.path.join(opts["restore_path"], 'ckpt', '*.index'))

            training_ckpts = sorted([c for c in ckpts if ckpt_pattern_idx.match(c)],
                                    key=lambda x: int(ckpt_pattern_idx.match(x).groups()[0]))

            weight_avg_ckpts = [c for c in ckpts if not ckpt_pattern_idx.match(c)]
            filenames = training_ckpts + weight_avg_ckpts
            filenames = [f[:-len(".index")] for f in filenames]
        else:
            filenames = sorted([f[:-len(".index")] for f in glob(opts['restore_path'] + '*.index')])

        possible_args = os.path.join(opts["restore_path"], 'arguments.json')
        if 'global_batch_size' in opts.keys():
            global_batch_size = opts['global_batch_size']
        elif os.path.isfile(possible_args):
            with open(os.path.join(opts["restore_path"], 'arguments.json'), 'r') as fp:
                try:
                    global_batch_size = json.load(fp)['global_batch_size']
                except KeyError:
                    global_batch_size = opts['micro_batch_size']
        else:
            global_batch_size = opts['micro_batch_size']
    else:
        filenames = [None]
        global_batch_size = opts['micro_batch_size']

    num_files = len(filenames)
    if num_files == 0:
        print(
            f"Error: no files found in --restore-path={opts['restore_path']}. "
            "Exiting."
        )
        sys.exit(1)

    if opts['use_popdist']:
        global_batch_size, num_files = valid.session.run(
            [valid.ops['broadcast_global_batch_size'],
             valid.ops['broadcast_num_files']],
            feed_dict={valid.placeholders['global_batch_size']: global_batch_size,
                       valid.placeholders['num_files']: num_files}
        )

    if opts['distributed_worker_index'] == 0 and not opts['generated_data']:
        print(filenames)

    total_samples = (opts['replicas'] *
                     opts['shards'] *
                     opts['validation_batches_per_step'] *
                     opts["validation_iterations"]
                     if opts['latency'] else 0)

    latency_thread = LatencyThread(valid, total_samples)
    success = False
    # Validation block
    logging.mlperf_logging(key="BLOCK_START", log_type="start",
                           metadata={"first_epoch_num": 1,
                                     "epoch_count": opts["epochs"]}
                           )

    for i in range(num_files):
        if opts['distributed_worker_index'] == 0:
            filename = filenames[i]
            if filename:
                print(filename)
                valid.saver.restore(valid.session, filename)
                pattern_match = ckpt_pattern.match(filename)
                if pattern_match:
                    iteration = int(pattern_match.groups()[0])
                else:
                    iteration = -1
            else:
                print("Warning: no restore point found - randomly initialising weights instead")
                valid.session.run(valid.init)
                iteration = 0
        else:
            iteration = 0

        if opts['use_popdist']:
            iteration = valid.session.run(valid.ops['broadcast_iteration'],
                                          feed_dict={valid.placeholders['iteration']: iteration})

        epoch = float(global_batch_size * iteration) / DATASET_CONSTANTS[opts['dataset']]['NUM_IMAGES']
        for r in range(opts["repeat"]):
            stats = validation_run(valid, None, iteration, epoch, i == 0, opts, latency_thread)
            # Handle skipped case
            if stats and "val_size" in stats and "val_acc" in stats:
                if stats["val_acc"] > MLPERF_EVAL_TARGET:
                    success = True

    logging.mlperf_logging(key="BLOCK_STOP", log_type="stop",
                           metadata={"first_epoch_num": 1}
                           )
    logging.mlperf_logging(key="RUN_STOP",
                           value={"success": success},
                           metadata={"epoch_num": round(epoch),
                                     "status": "success" if success else "aborted"})


def add_main_arguments(parser):
    group = parser.add_argument_group('Main')
    group.add_argument('--model', default='resnet', help="Choose model")
    group.add_argument('--restore-path', type=str,
                       help="Path to a single checkpoint to restore from or directory containing multiple checkpoints")
    group.add_argument('--repeat', type=int, default=1,
                       help="Repeat validation for debugging puposes")
    group.add_argument('--inference', type=bool, default=False,
                       help="""Run in inference mode, disabling accuracy all-reduce between replicas.
                               Useful for benchmarking.""")
    group.add_argument('--total-batch-size', type=int, default=None,
                       help="""When not specified global_batch_size becomes micro_batch_size.""")
    group.add_argument('--help', action='store_true', help='Show help information')
    return parser


def set_main_defaults(opts):
    opts['summary_str'] = "\n"


def set_validation_defaults(opts):
    if not opts['validation']:
        opts['summary_str'] += "No Validation\n"
    else:
        opts['validation_global_batch_size'] = opts['micro_batch_size']*opts['shards']*opts['replicas']*opts['distributed_worker_count']
        opts['summary_str'] += "Validation\n Batch Size: {}\n".format("{validation_global_batch_size}")
        opts["validation_iterations"] = (
            (int(DATASET_CONSTANTS[opts['dataset']]['NUM_VALIDATION_IMAGES']) + 128) //
            opts["validation_global_batch_size"]) + 1
        if opts["batches_per_step"] < opts["validation_iterations"]:
            opts["validation_batches_per_step"] = int(opts["validation_iterations"] //
                                                      int(round(opts["validation_iterations"] / opts['batches_per_step'])))
            opts["validation_iterations"] = int((opts["validation_iterations"] +
                                                 opts["validation_batches_per_step"] - 1) / opts["validation_batches_per_step"])
        else:
            opts["validation_batches_per_step"] = opts["validation_iterations"]
            opts["validation_iterations"] = 1


def create_parser(model, parser):
    parser = model.add_arguments(parser)
    parser = dataset.add_arguments(parser)
    parser = train.add_training_arguments(parser)
    parser = train.add_ipu_arguments(parser)
    parser = logging.add_arguments(parser)
    return parser


def set_distribution_defaults(opts):
    if opts['use_popdist']:
        opts['distributed_worker_count'] = popdist.getNumInstances()
        opts['distributed_worker_index'] = popdist.getInstanceIndex()
        opts['distributed_cluster'] = None

        opts['summary_str'] += 'Popdist\n'
        opts['summary_str'] += ' Process count: {distributed_worker_count}\n'
        opts['summary_str'] += ' Process index: {distributed_worker_index}\n'
    else:
        opts['distributed_worker_count'] = 1
        opts['distributed_worker_index'] = 0
        opts['distributed_cluster'] = None


def set_defaults(model, opts):
    set_main_defaults(opts)
    dataset.set_defaults(opts)
    model.set_defaults(opts)
    set_distribution_defaults(opts)
    set_validation_defaults(opts)
    train.set_ipu_defaults(opts)
    logging.set_defaults(opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validation for previously generated checkpoints.', add_help=False)
    parser = add_main_arguments(parser)
    parser = configurations.add_arguments(parser)
    args, unknown = parser.parse_known_args()
    args = configurations.parse_config(args, parser, known_args_only=True)
    args = vars(args)

    try:
        model = importlib.import_module("Models." + args['model'])
    except ImportError:
        raise ValueError('Models/{}.py not found'.format(args['model']))

    parser = create_parser(model, parser)
    opts = parser.parse_args()
    opts = configurations.parse_config(opts, parser)
    opts = vars(opts)

    if args['help']:
        parser.print_help()
    else:
        # backwards compatibility
        if opts['batch_size'] and opts['micro_batch_size']:
            raise ValueError('Both --batch-size and --micro-batch-size arguments were given, '
                             'use --micro-batch-size, as --batch-size is deprecated and kept '
                             'for backwards compatibility.')
        elif opts['batch_size']:
            opts['micro_batch_size'] = opts['batch_size']

        if popdist.isPopdistEnvSet():
            opts['use_popdist'] = True
            opts['replicas'] = popdist.getNumLocalReplicas()
            opts['total_replicas'] = popdist.getNumTotalReplicas()
            opts['select_ipu'] = str(popdist.getDeviceId())
        else:
            opts['use_popdist'] = False
            opts['total_replicas'] = opts['replicas']

        opts["command"] = ' '.join(sys.argv)
        set_defaults(model, opts)

        if opts['dataset'] == 'imagenet':
            if opts['image_size'] is None:
                opts['image_size'] = 224
        elif 'cifar' in opts['dataset']:
            opts['image_size'] = 32

        if opts["wandb"] and opts["distributed_worker_index"] == 0:
            logging.initialise_wandb(opts)
        logging.print_to_file_and_screen("Command line: " + opts["command"], opts)
        logging.print_to_file_and_screen(opts["summary_str"].format(**opts), opts)
        opts["summary_str"] = ""
        logging.print_to_file_and_screen(opts, opts)
        validation_only_process(model, opts)
