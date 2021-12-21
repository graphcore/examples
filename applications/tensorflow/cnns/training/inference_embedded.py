# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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
CNN inference using TF1 embedded application runtime.
See the README for more information.

"""

import tensorflow.compat.v1 as tf

import os
import re
import gc
import time
import json
import argparse
import sys
import uuid
import importlib
from collections import OrderedDict
from glob import glob
from threading import Thread
from queue import Queue

import numpy as np
from tensorflow.python import ipu
from tensorflow.python.ipu import loops, ipu_infeed_queue, ipu_outfeed_queue, \
                                  application_compile_op, embedded_runtime
from tensorflow.python.ipu.scopes import ipu_scope
import libpvti as pvti

import configurations
import log as logging
import train
from Datasets import data as dataset
from ipu_utils import get_config

DATASET_CONSTANTS = dataset.DATASET_CONSTANTS
MLPERF_EVAL_TARGET = 75.9


def get_exec_path(model, model_size, batch_size, batches_per_step, filenames, use_generated_data,
                  use_tmp_execs):
    """
    Return the path to compiled graph based on model parameters
    or random name if that was requested by setting an option.
    """
    if use_tmp_execs:
        random_part = str(uuid.uuid4())
        poplar_exec_filepath = f"model_{random_part}.poplar_exec"
    else:
        ckpt_name = filenames[0].split('/')[-1] if len(filenames) else 'random'
        label_phrase = "nolabels" if use_generated_data else "withlabels"
        poplar_exec_filepath = (f"model_{model}{model_size}_bs_{str(batch_size)}_"
                                f"bps_{str(batches_per_step)}_{str(ckpt_name)}_{label_phrase}"
                                f".poplar_exec")
    return poplar_exec_filepath


def validation_graph_builder(model, data_dict, opts):
    """Create a validation graph containing specified model and returning accuracy value."""
    train.ipuside_preprocessing(data_dict, opts, training=False)
    image = data_dict['image']
    logits = model(opts, training=False, image=image)
    probabilities = tf.reduce_max(logits, axis=1)
    predictions = tf.argmax(logits, 1, output_type=tf.int32)
    return probabilities, predictions


def get_ckpt_filenames(opts):
    """
    Look for a saved checkpoint based on specified execution options.
    Return checkpoints filenames (possibly more than 1) and calculated batch size.
    """
    ckpt_pattern_idx = re.compile(".*ckpt-([0-9]+).index$")
    if opts['restore_path'] and opts['distributed_worker_index'] == 0:
        if os.path.isdir(opts['restore_path']):
            # search to a maximum depth of 1
            ckpts = glob(os.path.join(opts['restore_path'], '*.index')) \
                    + glob(os.path.join(opts['restore_path'], 'ckpt', '*.index'))

            training_ckpts = sorted([c for c in ckpts if ckpt_pattern_idx.match(c)],
                                    key=lambda x: int(ckpt_pattern_idx.match(x).groups()[0]))

            weight_avg_ckpts = [c for c in ckpts if not ckpt_pattern_idx.match(c)]
            filenames = training_ckpts + weight_avg_ckpts
            filenames = [f[:-len(".index")] for f in filenames]
        else:
            filenames = sorted([f[:-len(".index")] for f in glob(opts['restore_path'] +
                                                                 '*.index')])

        possible_args = os.path.join(opts['restore_path'], 'arguments.json')
        if os.path.isfile(possible_args):
            with open(os.path.join(opts['restore_path'], 'arguments.json'), 'r') as fp:
                try:
                    total_batch_size = json.load(fp)['total_batch_size']
                except KeyError:
                    total_batch_size = opts['micro_batch_size']
        else:
            total_batch_size = opts['micro_batch_size']
    else:
        filenames = []
        total_batch_size = opts['micro_batch_size']

    return (filenames, total_batch_size)


def dataset_to_list(dataset, num_batches_to_process):
    """Convert specified number of samples from TF dataset to list of numpy arrays."""
    images = []
    labels = []
    index = 0
    dataset = tf.data.make_one_shot_iterator(dataset)
    with tf.Session() as sess:
        next_element = dataset.get_next()
        is_data = True
        while is_data and index < num_batches_to_process:
            try:
                output = sess.run(next_element)
                image = output['data_dict']['image']
                label = output['data_dict']['label']
                images.extend(image)
                labels.extend(label)
                index += 1
            except tf.errors.OutOfRangeError:
                is_data = False
    return images, labels


def prepare_feed_dict(placeholders, images, labels, batch_size, use_generated_data, idx=0):
    """Return the feed dict per batch with the input data."""
    idx = idx * batch_size
    if use_generated_data:
        return {
            placeholders[0]: images
        }
    return {
        placeholders[0]: images[idx:idx+batch_size],
        placeholders[1]: labels[idx:idx+batch_size]
    }


def configure_ipu(opts):
    """Set the IPU configuration based on execution options."""
    global_amp = None
    if opts['available_memory_proportion'] and len(opts['available_memory_proportion']) == 1:
        global_amp = opts['available_memory_proportion'][0]

    ipu_options = get_config(ipu_id=opts['select_ipu'],
                             prng=not opts['no_stochastic_rounding'],
                             shards=opts['shards'],
                             number_of_replicas=opts['total_replicas'],
                             max_cross_replica_buffer_size=opts['max_cross_replica_buffer_size'],
                             fp_exceptions=opts['fp_exceptions'],
                             half_partials=opts['enable_half_partials'],
                             conv_dithering=opts['enable_conv_dithering'],
                             enable_recomputation=opts['enable_recomputation'],
                             seed=opts['seed'],
                             availableMemoryProportion=global_amp,
                             stable_norm=opts['stable_norm'],
                             compile_only=opts['compile_only'],
                             internalExchangeOptimisationTarget=opts[
                                 "internal_exchange_optimisation_target"
                             ],
                             num_io_tiles=opts['num_io_tiles'],
                             number_of_distributed_batch_norm_replicas=opts.get("BN_span", 1),
                             nanoo=not opts['saturate_on_overflow'],
                             )

    if opts['on_demand']:
        ipu_options.device_connection.enable_remote_buffers = True
        ipu_options.device_connection.type = ipu.utils.DeviceConnectionType.ON_DEMAND

    ipu_options.configure_ipu_system()


def create_poplar_exec(model, opts, poplar_exec_path):
    """Create graph and save it to the file."""
    valid_graph = tf.Graph()

    with valid_graph.as_default():
        # datasets must be defined outside the ipu device scope
        if opts['generated_data']:
            # create dummy dataset with images only
            dummy_image = np.zeros((opts['micro_batch_size'], opts['image_size'],
                                    opts['image_size'], 3), dtype=np.uint8)
            inference_dataset = tf.data.Dataset.from_tensors({
                "image": dummy_image
            })
        else:
            # create dataset with images and labels
            inference_dataset = dataset.data(opts, is_training=False)
        inference_dataset = inference_dataset.map(lambda x: {'data_dict': x})

        inference_infeed_iterator = \
            ipu_infeed_queue.IPUInfeedQueue(inference_dataset,
                                            prefetch_depth=opts['prefetch_depth'])

        acc_queue = ipu_outfeed_queue.IPUOutfeedQueue()
        with ipu_scope('/device:IPU:0'):
            def comp_fn():
                def body(data_dict):
                    accuracy = validation_graph_builder(model, data_dict, opts)
                    accuracy_enqueue = acc_queue.enqueue(accuracy)
                    return accuracy_enqueue
                accuracy = loops.repeat(int(opts['validation_batches_per_step']),
                                        body, [], inference_infeed_iterator)
                return accuracy

        filenames, _ = get_ckpt_filenames(opts)

        accuracy = application_compile_op.experimental_application_compile_op(
            comp_fn, output_path=poplar_exec_path, freeze_variables=True)

        outfeed = acc_queue.dequeue()
        valid_saver = tf.train.Saver()

        ipu.utils.move_variable_initialization_to_cpu()

    with tf.Session(graph=valid_graph, config=tf.ConfigProto()) as sess:
        if len(filenames) == 1:
            print("Restoring from a snapshot: ", filenames[0])
            sess.run(inference_infeed_iterator.initializer)
            init = tf.global_variables_initializer()
            sess.run(init)
            valid_saver.restore(sess, filenames[0])
        else:
            print("Warning: no restore point found - randomly initialising weights instead")
            init = tf.global_variables_initializer()
            sess.run(init)

        path = sess.run(accuracy)
        print(f"Poplar executable: {path}")

    valid_graph.finalize()


def inference_run(exec_filename, ckpt_name, iteration, epoch, first_run, opts):
    """Run inference for multiple iterations and collect latency values."""
    logging.mlperf_logging(key="EVAL_START", log_type="start",
                           metadata={"epoch_num": round(epoch)})
    engine_name = "my_engine"
    ctx = embedded_runtime.embedded_runtime_start(exec_filename, [],
                                                  engine_name, timeout=1000)

    input_placeholder = tf.placeholder(tf.uint8, (opts['micro_batch_size'], opts['image_size'],
                                                  opts['image_size'], 3))

    num_iters = opts['iterations']
    if opts['generated_data']:
        placeholders = [input_placeholder]
        images = np.random.normal(size=(opts['micro_batch_size'], opts['image_size'],
                                        opts['image_size'], 3)).astype(np.uint8)
        labels = None
    else:
        label_placeholder = tf.placeholder(tf.int32, (opts['micro_batch_size']))
        placeholders = [input_placeholder, label_placeholder]

        with tf.Graph().as_default():
            inference_dataset = dataset.data(opts, is_training=False).map(lambda x:
                                                                          {'data_dict': x})
            images, labels = dataset_to_list(inference_dataset, num_iters * opts['micro_batch_size'])

    call_result = embedded_runtime.embedded_runtime_call(placeholders, ctx)

    ipu.config.reset_ipu_configuration()
    gc.collect()

    thread_queue = Queue()
    with tf.Session() as session:
        # do not include time of the first iteration in stats
        initial_feed_dict = prepare_feed_dict(placeholders, images, labels, opts['micro_batch_size'],
                                              opts['generated_data'], 0)
        session.run(call_result, initial_feed_dict)

        def runner(session, thread_idx):
            thread_channel = pvti.createTraceChannel(f"Thread {thread_idx}")
            latencies = []
            accuracies = []
            for iter_idx in range(num_iters):
                feed_dict = prepare_feed_dict(placeholders, images, labels, opts['micro_batch_size'],
                                              opts['generated_data'], iter_idx)
                with pvti.Tracepoint(thread_channel, f"Iteration {iter_idx}"):
                    start_iter = time.time()
                    _, predictions = session.run(call_result, feed_dict)
                    end_iter = time.time()
                latencies.append(end_iter - start_iter)
                if not opts['generated_data']:
                    expected = feed_dict[label_placeholder]
                    accuracy = np.mean(np.equal(predictions, expected).astype(np.float32))
                    accuracies.append(accuracy)
            thread_queue.put((latencies, accuracies), timeout=10)

        thp = [Thread(target=runner, args=(session, thread_idx))
               for thread_idx in range(opts['num_inference_thread'])]
        inference_start = time.time()
        for idx, _thread in enumerate(thp):
            _thread.start()
            print(f"Thread {idx} started")

        for idx, _thread in enumerate(thp):
            _thread.join()
            print(f"Thread {idx} joined")
        val_time = time.time() - inference_start

    latencies, accuracies = [], []
    while not thread_queue.empty():
        lat_acc = thread_queue.get()
        latencies.extend(lat_acc[0])
        accuracies.extend(lat_acc[1])

    if opts['generated_data']:
        total_accuracy = -1
    else:
        total_accuracy = sum(accuracies) / len(accuracies)
        total_accuracy *= 100

    # convert latencies to miliseconds
    latencies = [1000 * latency_s for latency_s in latencies]

    max_latency = max(latencies)
    mean_latency = np.mean(latencies)
    perc_99 = np.percentile(latencies, 99)
    perc_99_9 = np.percentile(latencies, 99.9)

    print(f"Latencies - avg: {mean_latency:8.4f}, 99th percentile: {perc_99:8.4f}, "
          f"99.9th percentile: {perc_99_9:8.4f}, max: {max_latency:8.4f}")

    valid_format = (
        "Validation top-1 accuracy [{name}] (iteration: {iteration:6d}, epoch: {epoch:6.2f}, "
        "img/sec: {img_per_sec:6.2f}, time: {val_time:8.6f}, "
        "latency (ms): {latency:8.4f}: {val_acc:6.3f}%")

    val_size = (num_iters * opts['num_inference_thread'] * opts['validation_total_batch_size'])

    stats = OrderedDict([
                ('name', ckpt_name),
                ('iteration', iteration),
                ('epoch', epoch),
                ('val_acc', total_accuracy),
                ('val_time', val_time),
                ('val_size', val_size),
                ('img_per_sec', val_size / val_time),
                ('latency', mean_latency),
            ])
    logging.print_to_file_and_screen(valid_format.format(**stats), opts)
    logging.write_to_csv(stats, first_run, False, opts)
    if opts['wandb'] and opts['distributed_worker_index'] == 0:
        logging.log_to_wandb(stats)
    logging.mlperf_logging(key="EVAL_STOP", log_type="stop",
                           metadata={"epoch_num": round(epoch)})
    logging.mlperf_logging(
        key="EVAL_ACCURACY", value=float(stats['val_acc'])/100,
        metadata={"epoch_num": round(epoch)})
    return stats


def inference_only_process(model, opts):
    """Create a graph (if there is no precompiled one) and run inference."""
    (filenames, total_batch_size) = get_ckpt_filenames(opts)

    poplar_exec_path = get_exec_path(opts['model'], opts['model_size'], opts['micro_batch_size'],
                                     opts['validation_batches_per_step'], filenames,
                                     opts['generated_data'], opts['tmp_execs'])
    configure_ipu(opts)
    if opts['force_recompile'] or opts['tmp_execs'] or not os.path.isfile(poplar_exec_path):
        create_poplar_exec(model.Model, opts, poplar_exec_path)

    # Validation block
    logging.mlperf_logging(key="BLOCK_START", log_type="start",
                           metadata={"first_epoch_num": 1,
                                     "epoch_count": opts['epochs']})

    ckpt_name = filenames[0].split('/')[-1] if len(filenames) else 'random'
    ckpt_pattern = re.compile(".*ckpt-([0-9]+)$")
    pattern_match = ckpt_pattern.match(ckpt_name)
    if pattern_match:
        iteration = int(pattern_match.groups()[0])
    else:
        iteration = -1
    epoch = float(total_batch_size * iteration) / DATASET_CONSTANTS[opts['dataset']]['NUM_IMAGES']

    success = False
    for _ in range(opts['repeat']):
        stats = inference_run(poplar_exec_path, ckpt_name, iteration, epoch, True, opts)
        # Handle skipped case
        if stats and "val_size" in stats and "val_acc" in stats:
            if stats['val_acc'] > MLPERF_EVAL_TARGET:
                success = True

    logging.mlperf_logging(key="BLOCK_STOP", log_type="stop",
                           metadata={"first_epoch_num": 1})
    logging.mlperf_logging(key="RUN_STOP",
                           value={"success": success},
                           metadata={"epoch_num": round(epoch),
                                     "status": "success" if success else "aborted"})

    if opts['tmp_execs'] and os.path.isfile(poplar_exec_path):
        os.remove(poplar_exec_path)


def add_main_arguments(parser):
    group = parser.add_argument_group('Main')
    group.add_argument('--model', default='resnet', help="Choose model")
    group.add_argument('--restore-path', type=str,
                       help="Path to a single checkpoint to restore from or directory "
                            "containing multiple checkpoints")
    group.add_argument('--repeat', type=int, default=1,
                       help="Repeat inference for debugging puposes")
    group.add_argument('--num-inference-thread', type=int, default=1,
                       help="Number of inference threads to run in parallel")
    group.add_argument('--force-recompile', action='store_true', default=False,
                       help='Force application to recompile graph even if there '
                            'is one already on the drive')
    group.add_argument('--tmp-execs', action='store_true', default=False,
                       help='Save compiled executable with random name and remove it after '
                            'the inference.')
    group.add_argument('--help', action='store_true', help='Show help information')
    return parser


def set_main_defaults(opts):
    opts['summary_str'] = "\n"


def set_validation_defaults(opts):
    opts['validation_total_batch_size'] = opts['micro_batch_size'] * opts['shards'] * \
                                            opts['replicas'] * opts['distributed_worker_count']
    opts['summary_str'] += f"Validation\n Batch Size: {opts['validation_total_batch_size']}\n"
    opts['validation_iterations'] = opts['iterations']
    opts['validation_batches_per_step'] = opts['batches_per_step']


def create_parser(model, parser):
    parser = model.add_arguments(parser)
    parser = dataset.add_arguments(parser)
    parser = train.add_training_arguments(parser)
    parser = train.add_ipu_arguments(parser)
    parser = logging.add_arguments(parser)
    return parser


def set_distribution_defaults(opts):
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
    parser = argparse.ArgumentParser(description="Validation for previously generated "
                                                 "checkpoints.", add_help=False)
    parser = add_main_arguments(parser)
    parser = configurations.add_arguments(parser)
    args, unknown = parser.parse_known_args()
    args = configurations.parse_config(args, parser, known_args_only=True)
    args = vars(args)

    try:
        model = importlib.import_module("Models." + args['model'])
    except ImportError as ie:
        raise ValueError(f"Models/{args['model']}.py not found") from ie

    parser = create_parser(model, parser)
    opts = parser.parse_args()
    opts = configurations.parse_config(opts, parser)
    opts = vars(opts)
    print(opts)

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
        opts['use_popdist'] = False
        opts['total_replicas'] = opts['replicas']

        opts['command'] = ' '.join(sys.argv)
        set_defaults(model, opts)

        if opts['dataset'] == 'imagenet':
            if opts['image_size'] is None:
                opts['image_size'] = 224
        elif 'cifar' in opts['dataset']:
            opts['image_size'] = 32

        if opts['wandb'] and opts['distributed_worker_index'] == 0:
            logging.initialise_wandb(opts)
        logging.print_to_file_and_screen("Command line: " + opts['command'], opts)
        logging.print_to_file_and_screen(opts['summary_str'].format(**opts), opts)
        opts['summary_str'] = ""
        logging.print_to_file_and_screen(opts, opts)
        inference_only_process(model, opts)
