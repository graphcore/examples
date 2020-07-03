# Copyright 2019 Graphcore Ltd.
"""
The validation code used in train.py.

This script can also be called to run validation on previously generated checkpoints.
See the README for more information.

"""

import tensorflow as tf
import os
import re
import time
import argparse
import sys
from collections import OrderedDict
import importlib
from glob import glob

import train
import log as logging
from Datasets import data as dataset
from tensorflow.python import ipu
from ipu_utils import get_config
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import loops, ipu_infeed_queue
import tensorflow.contrib.compiler.xla as xla
from tensorflow.python.ipu.ops import cross_replica_ops
DATASET_CONSTANTS = dataset.DATASET_CONSTANTS


def validation_graph_builder(model, image, label, opts):
    logits = model(opts, training=False, image=image)

    predictions = tf.argmax(logits, 1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, label), tf.float16))

    return accuracy


def validation_graph(model, opts):
    valid_graph = tf.Graph()
    with valid_graph.as_default():
        # datasets must be defined outside the ipu device scope
        valid_iterator = ipu_infeed_queue.IPUInfeedQueue(dataset.data(opts, is_training=False),
                                                         feed_name='validation_feed',
                                                         replication_factor=opts['replicas']*opts['shards'])

        with ipu_scope('/device:IPU:0'):
            def comp_fn():
                def body(total_accuracy, image, label):
                    accuracy = validation_graph_builder(model, image, label, opts)
                    return total_accuracy + (tf.cast(accuracy, tf.float32) / opts["validation_batches_per_step"])
                accuracy = loops.repeat(int(opts["validation_batches_per_step"]),
                                        body, [tf.constant(0, tf.float32)], valid_iterator)
                if opts['replicas'] > 1:
                    accuracy = cross_replica_ops.cross_replica_sum(accuracy) / (opts['replicas']*opts['shards'])
                return accuracy

            (accuracy,) = xla.compile(comp_fn, [])

        accuracy = 100 * accuracy

        valid_saver = tf.train.Saver()

        ipu.utils.move_variable_initialization_to_cpu()
        valid_init = tf.global_variables_initializer()

    globalAMP = None
    if opts["available_memory_proportion"] and len(opts["available_memory_proportion"]) == 1:
        globalAMP = opts["available_memory_proportion"][0]

    ipu_options = get_config(ipu_id=opts["select_ipu"],
                             prng=not opts["no_stochastic_rounding"],
                             shards=1,
                             number_of_replicas=opts['replicas']*opts['shards'],
                             max_cross_replica_buffer_size=opts["max_cross_replica_buffer_size"],
                             fp_exceptions=opts["fp_exceptions"],
                             xla_recompute=opts["xla_recompute"],
                             seed=opts["seed"],
                             profile = opts['profile'],
                             availableMemoryProportion=globalAMP,
                             stable_norm=opts["stable_norm"])
    ipu.utils.configure_ipu_system(ipu_options)

    valid_sess = tf.Session(graph=valid_graph, config=tf.ConfigProto())

    return train.GraphOps(valid_graph, valid_sess, valid_init, [accuracy], None, valid_iterator, None, valid_saver, None)


def validation_run(valid, filepath, i, epoch, first_run, opts):
    if filepath:
        valid.saver.restore(valid.session, filepath)
        name = filepath.split('/')[-1]
    else:
        name = None

    # Gather accuracy statistics
    accuracy = 0.0
    start = time.time()
    for __ in range(opts["validation_iterations"]):
        try:
            a = valid.session.run(valid.ops)[0]
        except tf.errors.OpError as e:
            raise tf.errors.ResourceExhaustedError(e.node_def, e.op, e.message)

        accuracy += a
    val_time = time.time() - start
    accuracy /= opts["validation_iterations"]

    valid_format = (
        "Validation top-1 accuracy [{name}] (iteration: {iteration:6d}, epoch: {epoch:6.2f}, img/sec: {img_per_sec:6.2f},"
        " latency (ms): {latency:8.4g}, time: {val_time:8.6f}): {val_acc:6.3f}%")

    stats = OrderedDict([
                ('name', name),
                ('iteration', i),
                ('epoch', epoch),
                ('val_acc', accuracy),
                ('val_time', val_time),
                ('img_per_sec', (opts["validation_iterations"] *
                                 opts["validation_batches_per_step"] *
                                 opts['validation_total_batch_size']) / val_time),
                ('latency', 1000 * val_time / (opts["validation_iterations"] *
                                               opts["validation_batches_per_step"])),
            ])
    logging.print_to_file_and_screen(valid_format.format(**stats), opts)
    del stats['name']
    logging.write_to_csv(stats, first_run, False, opts)


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

    filename_pattern = re.compile(".*ckpt-[0-9]+$")
    ckpt_pattern = re.compile(".*ckpt-([0-9]+)$")
    if opts["restore_path"]:
        if os.path.isdir(opts["restore_path"]):
            filenames = sorted([os.path.join(opts["restore_path"], f[:-len(".index")])
                                for f in os.listdir(opts["restore_path"])
                                if filename_pattern.match(f[:-len(".index")]) and
                                f[-len(".index"):] == ".index"],
                               key=lambda x: int(ckpt_pattern.match(x).groups()[0]))
        else:
            filenames = sorted([f[:-len(".index")] for f in glob(opts['restore_path'] + '*.index')])
    else:
        filenames = [None]

    print(filenames)

    for i, filename in enumerate(filenames):
        print(filename)
        if filename:
            valid.saver.restore(valid.session, filename)
            if ckpt_pattern.match(filename):
                iteration = int(ckpt_pattern.match(filename).groups()[0])
            else:
                iteration = -1
        else:
            print("Warning: no restore point found - randomly initialising weights instead")
            valid.session.run(valid.init)
            iteration = 0

        epoch = float(opts["batch_size"] * iteration) / DATASET_CONSTANTS[opts['dataset']]['NUM_IMAGES']
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
        opts['validation_total_batch_size'] = opts['batch_size']*opts['shards']*opts['replicas']
        opts['summary_str'] += "Validation\n Batch Size: {}\n".format("{validation_total_batch_size}")
        opts["validation_iterations"] = int(DATASET_CONSTANTS[opts['dataset']]['NUM_VALIDATION_IMAGES'] /
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


def set_defaults(model, opts):
    set_main_defaults(opts)
    dataset.set_defaults(opts)
    model.set_defaults(opts)
    set_validation_defaults(opts)
    train.set_ipu_defaults(opts)
    logging.set_defaults(opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validation for previously generated checkpoints.', add_help=False)
    parser = add_main_arguments(parser)
    args, unknown = parser.parse_known_args()
    args = vars(args)
    if args['help']:
        parser.print_help()
    else:
        try:
            model = importlib.import_module("Models." + args['model'])
        except ImportError:
            raise ValueError('Models/{}.py not found'.format(args['model']))

        parser = create_parser(model, parser)
        opts = vars(parser.parse_args())
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
