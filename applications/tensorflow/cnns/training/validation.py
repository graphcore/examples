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

import train
import log as logging
from Datasets import data as dataset
from Datasets.imagenet_dataset import accelerator_side_preprocessing
from tensorflow.python import ipu
from ipu_utils import get_config
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import loops, ipu_infeed_queue
import tensorflow.contrib.compiler.xla as xla
from tensorflow.python.ipu.ops import cross_replica_ops
from tensorflow.python.ipu import horovod as hvd
import popdist
import popdist.tensorflow
import configurations

DATASET_CONSTANTS = dataset.DATASET_CONSTANTS
config_file = Path(Path(__file__).parent, Path("configs.yml"))


def validation_graph_builder(model, data_dict, opts):
    if opts['dataset'] == 'imagenet' and not opts.get('hostside_norm'):
        data_dict['image'] = accelerator_side_preprocessing(data_dict['image'], opts=opts)

    image, label = data_dict['image'], data_dict['label']
    logits = model(opts, training=False, image=image)
    predictions = tf.argmax(logits, 1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, label), tf.float16))
    return accuracy


def validation_graph(model, opts):

    if opts['use_popdist']:
        hvd.init()

    valid_graph = tf.Graph()
    with valid_graph.as_default():
        # datasets must be defined outside the ipu device scope
        valid_dataset = dataset.data(opts, is_training=False).map(lambda x: {'data_dict': x})

        valid_iterator = ipu_infeed_queue.IPUInfeedQueue(valid_dataset,
                                                         feed_name='validation_feed',
                                                         replication_factor=opts['replicas']*opts['shards'])

        with ipu_scope('/device:IPU:0'):
            def comp_fn():
                def body(total_accuracy, data_dict):
                    accuracy = validation_graph_builder(model, data_dict, opts)
                    return total_accuracy + (tf.cast(accuracy, tf.float32) / opts["validation_batches_per_step"])
                accuracy = loops.repeat(int(opts["validation_batches_per_step"]),
                                        body, [tf.constant(0, tf.float32)], valid_iterator)
                if opts['total_replicas']*opts['shards'] > 1:
                    accuracy = cross_replica_ops.cross_replica_sum(accuracy) / (opts['total_replicas']*opts['shards'])
                return accuracy

            (accuracy,) = xla.compile(comp_fn, [])

        accuracy = 100 * accuracy

        valid_saver = tf.train.Saver()

        ipu.utils.move_variable_initialization_to_cpu()
        valid_init = tf.global_variables_initializer()

        if opts['use_popdist']:
            broadcast_ops = []
            for var in tf.global_variables():
                broadcast_ops.append(var.assign(hvd.broadcast(var, root_rank=0)))
        else:
            broadcast_ops = None

    globalAMP = None
    if opts["available_memory_proportion"] and len(opts["available_memory_proportion"]) == 1:
        globalAMP = opts["available_memory_proportion"][0]

    ipu_options = get_config(ipu_id=opts["select_ipu"],
                             prng=not opts["no_stochastic_rounding"],
                             shards=opts['shards'],
                             number_of_replicas=opts['replicas'],
                             max_cross_replica_buffer_size=opts["max_cross_replica_buffer_size"],
                             fp_exceptions=opts["fp_exceptions"],
                             half_partials=opts["enable_half_partials"],
                             conv_dithering=opts["enable_conv_dithering"],
                             xla_recompute=opts["xla_recompute"],
                             seed=opts["seed"],
                             profile = opts['profile'],
                             availableMemoryProportion=globalAMP,
                             stable_norm=opts["stable_norm"],
                             internalExchangeOptimisationTarget=opts[
                                 "internal_exchange_optimisation_target"
                             ],
                             limitVertexState=opts.get("limitVertexState", True))

    if opts['use_popdist']:
        ipu_options = popdist.tensorflow.set_ipu_config(ipu_options, opts['shards'], configure_device=False)

    ipu.utils.configure_ipu_system(ipu_options)

    valid_sess = tf.Session(graph=valid_graph, config=tf.ConfigProto())

    return train.GraphOps(valid_graph, valid_sess, valid_init, [accuracy, broadcast_ops], None, valid_iterator, None, valid_saver, None)


def validation_run(valid, filepath, i, epoch, first_run, opts):
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

    if opts['use_popdist']:
        # synchronise the model weights across all instances
        valid.session.run(valid.ops[1])

    if run:
        # Gather accuracy statistics
        accuracy = 0.0
        start = time.time()
        for __ in range(opts["validation_iterations"]):
            try:
                a = valid.session.run(valid.ops[0])
            except tf.errors.OpError as e:
                raise tf.errors.ResourceExhaustedError(e.node_def, e.op, e.message)

            accuracy += a
        val_time = time.time() - start
        accuracy /= opts["validation_iterations"]

        valid_format = (
            "Validation top-1 accuracy [{name}] (iteration: {iteration:6d}, epoch: {epoch:6.2f}, img/sec: {img_per_sec:6.2f},"
            " time: {val_time:8.6f}): {val_acc:6.3f}%")

        val_size = (opts["validation_iterations"] *
                    opts["validation_batches_per_step"] *
                    opts["validation_total_batch_size"])

        stats = OrderedDict([
                    ('name', name),
                    ('iteration', i),
                    ('epoch', epoch),
                    ('val_acc', accuracy),
                    ('val_time', val_time),
                    ('val_size', val_size),
                    ('img_per_sec', val_size / val_time),
                ])
        logging.print_to_file_and_screen(valid_format.format(**stats), opts)
        logging.write_to_csv(stats, first_run, False, opts)
        return stats


def initialise_validation(model, opts):
    # -------------- BUILD GRAPH ------------------
    valid = validation_graph(model.Model, opts)
    # ------------- INITIALIZE SESSION -----------

    valid.session.run(valid.iterator.initializer)
    with valid.graph.as_default():
        valid.session.run(tf.global_variables_initializer())

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
        if os.path.isfile(possible_args):
            with open(os.path.join(opts["restore_path"], 'arguments.json'), 'r') as fp:
                total_batch_size = json.load(fp)['total_batch_size']
        else:
            total_batch_size = opts['batch_size']
    else:
        filenames = [None]
        total_batch_size = opts['batch_size']

    num_files = len(filenames)

    if opts['use_popdist']:
        with tf.Graph().as_default(), tf.Session():
            # synchronise total_batch_size across instances
            local_tensor = tf.constant(total_batch_size)
            root_tensor = hvd.broadcast(local_tensor, root_rank=0)
            total_batch_size = root_tensor.eval()

            # synchonise num_files across instances
            local_tensor = tf.constant(num_files)
            root_tensor = hvd.broadcast(local_tensor, root_rank=0)
            num_files = root_tensor.eval()

    if opts['distributed_worker_index'] == 0:
        print(filenames)

    for i in range(num_files):
        if opts['distributed_worker_index'] == 0:
            filename = filenames[i]
            print(filename)
            if filename:
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
            with tf.Graph().as_default(), tf.Session():
                # synchronise iteration across instances
                local_tensor = tf.constant(iteration)
                root_tensor = hvd.broadcast(local_tensor, root_rank=0)
                iteration = root_tensor.eval()

        epoch = float(total_batch_size * iteration) / DATASET_CONSTANTS[opts['dataset']]['NUM_IMAGES']
        for r in range(opts["repeat"]):
            validation_run(valid, None, iteration, epoch, i == 0, opts)


def add_main_arguments(parser):
    group = parser.add_argument_group('Main')
    group.add_argument('--model', default='resnet', help="Choose model")
    group.add_argument('--restore-path', type=str,
                       help="Path to a single checkpoint to restore from or directory containing multiple checkpoints")
    group.add_argument('--repeat', type=int, default=1,
                       help="Repeat validation for debugging puposes")
    group.add_argument('--help', action='store_true', help='Show help information')
    return parser


def set_main_defaults(opts):
    opts['summary_str'] = "\n"


def set_validation_defaults(opts):
    if not opts['validation']:
        opts['summary_str'] += "No Validation\n"
    else:
        opts['validation_total_batch_size'] = opts['batch_size']*opts['shards']*opts['replicas']*opts['distributed_worker_count']
        opts['summary_str'] += "Validation\n Batch Size: {}\n".format("{validation_total_batch_size}")
        opts["validation_iterations"] = int(DATASET_CONSTANTS[opts['dataset']]['NUM_VALIDATION_IMAGES'] //
                                            opts["validation_total_batch_size"])
        if opts["batches_per_step"] < opts["validation_iterations"]:
            opts["validation_batches_per_step"] = int(opts["validation_iterations"] //
                                                      int(round(opts["validation_iterations"] / opts['batches_per_step'])))
            opts["validation_iterations"] = int(opts["validation_iterations"] / opts["validation_batches_per_step"])
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
    parser = configurations.add_arguments(parser, config_file)
    args, unknown = parser.parse_known_args()
    args = configurations.parse_config(args, parser, config_file, known_args_only=True)
    args = vars(args)
    if args['help']:
        parser.print_help()
    else:
        try:
            model = importlib.import_module("Models." + args['model'])
        except ImportError:
            raise ValueError('Models/{}.py not found'.format(args['model']))

        parser = create_parser(model, parser)
        opts = parser.parse_args()
        opts = configurations.parse_config(opts, parser, config_file)
        opts = vars(opts)
        print(opts)
        opts = vars(parser.parse_args())

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

        logging.print_to_file_and_screen("Command line: " + opts["command"], opts)
        logging.print_to_file_and_screen(opts["summary_str"].format(**opts), opts)
        opts["summary_str"] = ""
        logging.print_to_file_and_screen(opts, opts)
        validation_only_process(model, opts)
