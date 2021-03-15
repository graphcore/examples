#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import time
import argparse
import datetime
import random
from socket import gethostname
from collections import OrderedDict, namedtuple, Counter
from shutil import copytree
from contextlib import ExitStack
import json
import numpy as np
import sys
import math
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
from tensorflow.python.ipu import pipelining_ops

from ipu_optimizer import get_optimizer
import ipu_utils
import log
import logging
from log import logger
from functools import partial
import Datasets.data_loader as dataset
import modeling as bert_ipu
from multi_stage_wrapper import get_split_embedding_stages, get_split_matmul_stages, MultiStageEmbedding
from lr_schedules import make_lr_schedule
from loss_scaling_schedule import LossScalingScheduler
from poplar_options import set_poplar_engine_options
from tensorflow.python.ipu import horovod as hvd

import popdist
import popdist.tensorflow

GraphOps = namedtuple('graphOps',
                      ['graph', 'session', 'init', 'ops', 'placeholders',
                       'iterator', 'outfeed', 'saver', 'restore', 'tvars'])


def create_popdist_strategy():
    """
    Creates a distribution strategy for use with popdist. We use the
    Horovod-based IPUMultiReplicaStrategy. Horovod is used for the initial
    broadcast of the weights and when reductions are requested on the host.
    Imports are placed here so they are only done when required, as Horovod
    might not always be available.
    """

    from tensorflow.python.ipu.horovod import ipu_multi_replica_strategy

    hvd.init()

    # We add the IPU cross replica reductions explicitly in the IPUOptimizer,
    # so disable them in the IPUMultiReplicaStrategy.
    return ipu_multi_replica_strategy.IPUMultiReplicaStrategy(
        add_ipu_cross_replica_reductions=False)


def build_pretrain_pipeline_stages(model, bert_config, opts):
    """
    build pipeline stages according to "pipeline_stages" in config file
    """

    # flatten stages config into list of layers
    flattened_layers = []
    for stage in opts['pipeline_stages']:
        flattened_layers.extend(stage)
    layer_counter = Counter(flattened_layers)
    assert layer_counter['hid'] == opts['num_hidden_layers']
    assert layer_counter['emb'] == layer_counter['mlm']
    # gradient_accumulation_count needs to be a multiple of stage_number*2
    # this is constrained by sdk
    assert opts['gradient_accumulation_count'] % (len(opts['pipeline_stages'])*2) == 0

    computational_stages = []
    if layer_counter['emb'] > 1:
        # support distribute embedding to multiple IPUs
        embedding = MultiStageEmbedding(embedding_size=bert_config.hidden_size,
                                        vocab_size=bert_config.vocab_size,
                                        initializer_range=bert_config.initializer_range,
                                        n_stages=layer_counter['emb'],
                                        matmul_serialize_factor=opts["matmul_serialize_factor"],
                                        dtype=bert_config.dtype)
        embedding_stages = get_split_embedding_stages(
            embedding=embedding, split_count=layer_counter['emb'], bert_config=bert_config, batch_size=opts["batch_size"], seq_length=opts['seq_length'])
        # masked lm better be on same ipu with embedding layer for saving storage
        masked_lm_output_post_stages = get_split_matmul_stages(
            embedding=embedding, split_count=layer_counter['emb'], bert_config=bert_config)
    else:
        embedding_stages = [model.embedding_lookup_layer]
        masked_lm_output_post_stages = [model.mlm_head]

    layers = {
        'emb': embedding_stages,
        'pos': model.embedding_postprocessor_layer,
        'hid': model.encoder,
        'mlm': masked_lm_output_post_stages,
        'nsp': model.get_next_sentence_output_layer
    }
    stage_layer_list = []
    for stage in opts['pipeline_stages']:
        func_list = []
        for layer in stage:
            # embedding layer and mlm layer can be splited to mutliple IPUs, so need to be dealt with separately
            if layer == 'emb':
                func_list.append(embedding_stages[0])
                embedding_stages = embedding_stages[1:]
            elif layer == 'mlm':
                func_list.append(masked_lm_output_post_stages[0])
                masked_lm_output_post_stages = masked_lm_output_post_stages[1:]
            else:
                func_list.append(layers[layer])
        stage_layer_list.append(func_list)
    computational_stages = ipu_utils.stages_constructor(
        stage_layer_list, ['learning_rate', 'loss_scaling'],
        ['learning_rate', 'loss_scaling', 'mlm_loss', 'nsp_loss', 'mlm_acc', 'nsp_acc'])

    return computational_stages


def build_network(infeed,
                  outfeed,
                  bert_config=None,
                  opts=None,
                  learning_rate=None,
                  loss_scaling=None,
                  is_training=True):

    # build model
    pipeline_model = bert_ipu.BertModel(bert_config, is_training=is_training)

    # build stages & device mapping
    computational_stages = build_pretrain_pipeline_stages(
        pipeline_model, bert_config, opts,)
    device_mapping = opts['device_mapping']

    logger.info(
        f"************* computational stages: *************\n{computational_stages}")
    logger.info(
        f"************* device mapping: *************\n{device_mapping}")

    # define optimizer
    def optimizer_function(learning_rate, loss_scaling, mlm_loss, nsp_loss, mlm_acc, nsp_acc):
        total_loss = mlm_loss + nsp_loss
        optimizer = get_optimizer(learning_rate, loss_scaling, opts['total_replicas'], opts)
        return ipu.ops.pipelining_ops.OptimizerFunctionOutput(optimizer, total_loss*loss_scaling)

    # Set IPU-specific available memory proportion
    if isinstance(opts['available_memory_proportion'], float):
        available_memory_proportion_list = [
            str(opts['available_memory_proportion'])
        ] * len(device_mapping)
    else:
        available_memory_proportion_list = [
            str(opts['available_memory_proportion'][device]) for device in device_mapping
        ]

    if len(available_memory_proportion_list) != len(device_mapping):
        raise ValueError(
            "The available_memory_proportion list must be the same length as the number of stages in the pipeline."
        )

    options = [ipu.pipelining_ops.PipelineStageOptions(
        matmul_options={
            "availableMemoryProportion": amp,
            "partialsType": opts["partials_type"]
        }) for amp in available_memory_proportion_list
    ]

    # define pipeline schedule
    pipeline_schedule = pipelining_ops.PipelineSchedule.Grouped
    # TODO (nicolasc): I don't think this is supported for BERT as we have multiple stages on the same IPU
    if opts["pipeline_schedule"] == "Interleaved":
        pipeline_schedule = pipelining_ops.PipelineSchedule.Interleaved

    if is_training:
        pipeline_ops = ipu.ops.pipelining_ops.pipeline(computational_stages=computational_stages,
                                                       gradient_accumulation_count=int(
                                                           opts['gradient_accumulation_count']),
                                                       repeat_count=opts['batches_per_step'],
                                                       inputs=[learning_rate, loss_scaling],
                                                       infeed_queue=infeed,
                                                       outfeed_queue=outfeed,
                                                       optimizer_function=optimizer_function,
                                                       device_mapping=device_mapping,
                                                       forward_propagation_stages_poplar_options=options,
                                                       backward_propagation_stages_poplar_options=options,
                                                       offload_weight_update_variables=opts["variable_offloading"],
                                                       pipeline_schedule=pipeline_schedule,
                                                       recomputation_mode=ipu.ops.pipelining_ops.RecomputationMode[
                                                           opts['recomputation_mode']],
                                                       name="Pipeline")
    else:
        pipeline_ops = ipu.ops.pipelining_ops.pipeline(computational_stages=computational_stages,
                                                       gradient_accumulation_count=int(
                                                           opts['gradient_accumulation_count']),
                                                       repeat_count=opts['batches_per_step'],
                                                       inputs=[learning_rate, loss_scaling],
                                                       infeed_queue=infeed,
                                                       outfeed_queue=outfeed,
                                                       device_mapping=device_mapping,
                                                       forward_propagation_stages_poplar_options=options,
                                                       backward_propagation_stages_poplar_options=options,
                                                       offload_weight_update_variables=opts["variable_offloading"],
                                                       pipeline_schedule=pipeline_schedule,
                                                       recomputation_mode=ipu.ops.pipelining_ops.RecomputationMode[
                                                           opts['recomputation_mode']],
                                                       name="Pipeline")

    return pipeline_ops


def distributed_per_replica(function):
    """Run the function with the distribution strategy (if any) in a per-replica context."""
    def wrapper(*arguments):
        if tf.distribute.has_strategy():
            strategy = tf.distribute.get_strategy()
            return strategy.experimental_run_v2(function, args=arguments)
        else:
            return function(*arguments)

    return wrapper


@distributed_per_replica
def training_step_with_infeeds_and_outfeeds(train_iterator, outfeed_queue, bert_config, opts, learning_rate, loss_scaling, is_training):
    """
    Training step that uses an infeed loop with outfeeds. This runs 'iterations_per_step' steps per session call. This leads to
    significant speed ups on IPU. Not compatible with running on CPU or GPU.
    """

    if opts['gradient_accumulation_count'] > 1:
        training_step = partial(build_network,
                                infeed=train_iterator,
                                outfeed=outfeed_queue,
                                bert_config=bert_config,
                                opts=opts,
                                learning_rate=learning_rate,
                                loss_scaling=loss_scaling,
                                is_training=is_training)

    return ipu.ipu_compiler.compile(training_step, [])


def build_graph(opts, is_training=True, feed_name=None):
    train_graph = tf.Graph()
    strategy = None

    if opts['use_popdist']:
        strategy = create_popdist_strategy()

    with train_graph.as_default(), ExitStack() as stack:
        if strategy:
            stack.enter_context(strategy.scope())

        bert_config = bert_ipu.BertConfig.from_dict(opts)
        bert_config.dtype = tf.float32 if opts["precision"] == '32' else tf.float16

        # define placeholders
        placeholders = {
            'learning_rate': tf.placeholder(bert_config.dtype, shape=[]),
            'loss_scaling': tf.placeholder(bert_config.dtype, shape=[])
        }
        learning_rate = placeholders['learning_rate']
        loss_scaling = placeholders['loss_scaling']

        # define input, datasets must be defined outside the ipu device scope.
        train_iterator = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset.load(opts, is_training=is_training),
                                                             feed_name=feed_name+"_in", replication_factor=opts['replicas'])
        # define output
        outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
            feed_name=feed_name+"_out", replication_factor=opts['replicas'])

        # building networks with pipeline
        def bert_net():
            return build_network(train_iterator,
                                 outfeed_queue,
                                 bert_config,
                                 opts,
                                 learning_rate,
                                 loss_scaling,
                                 is_training)

        with ipu.scopes.ipu_scope('/device:IPU:0'):
            train = training_step_with_infeeds_and_outfeeds(train_iterator,
                                                            outfeed_queue,
                                                            bert_config,
                                                            opts,
                                                            learning_rate,
                                                            loss_scaling,
                                                            is_training)

        # get result from outfeed queue
        outfeed = outfeed_queue.dequeue()

        if strategy:
            # Take the mean of all the outputs across the distributed workers
            outfeed = [strategy.reduce(tf.distribute.ReduceOp.MEAN, v) for v in outfeed]

        if opts['distributed_worker_index'] == 0 or opts['log_all_workers']:
            log.print_trainable_variables(opts)

        model_and_optimiser_variables = tf.global_variables()
        model_variables = tf.trainable_variables() + tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)
        restore = tf.train.Saver(
            var_list=model_and_optimiser_variables
            if opts['restore_optimiser_from_checkpoint'] else model_variables)

        train_saver = tf.train.Saver(
            var_list=model_and_optimiser_variables
            if opts['save_optimiser_to_checkpoint'] else model_variables,
            max_to_keep=5)

        ipu.utils.move_variable_initialization_to_cpu()
        train_init = tf.global_variables_initializer()
        tvars = tf.trainable_variables()

    # calculate the number of required IPU
    num_ipus = (max(opts['device_mapping']) + 1) * opts['replicas']
    num_ipus = ipu_utils.next_power_of_two(num_ipus)

    ipu_options = ipu_utils.get_config(
        fp_exceptions=opts["fp_exceptions"],
        xla_recompute=opts["xla_recompute"],
        disable_graph_outlining=False,
        num_required_ipus=num_ipus,
        enable_stochastic_rounding=opts['stochastic_rounding'],
        max_cross_replica_sum_buffer_size=opts['max_cross_replica_sum_buffer_size'],
        scheduler_selection=opts['scheduler'],
        compile_only=opts['compile_only'],
        ipu_id = opts['select_ipu'])

    if opts['use_popdist']:
        ipu_options = popdist.tensorflow.set_ipu_config(ipu_options, opts['shards'], configure_device=False)

    ipu.utils.configure_ipu_system(ipu_options)

    # This is a workaround bug https://github.com/tensorflow/tensorflow/issues/23780
    from tensorflow.core.protobuf import rewriter_config_pb2
    sess_cfg = tf.ConfigProto()
    sess_cfg.graph_options.rewrite_options.memory_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF)

    train_sess = tf.Session(graph=train_graph, config=sess_cfg)

    return GraphOps(train_graph, train_sess, train_init, [train], placeholders, train_iterator, outfeed, train_saver, restore, tvars)


def training_step(train, learning_rate, loss_scaling):
    start = time.time()
    _ = train.session.run(train.ops, feed_dict={
                          train.placeholders['learning_rate']: learning_rate,
                          train.placeholders['loss_scaling']: loss_scaling})
    batch_time = (time.time() - start)
    if not os.environ.get('TF_POPLAR_FLAGS') or '--use_synthetic_data' not in os.environ.get('TF_POPLAR_FLAGS'):
        _learning_rate, _loss_scaling_, _mlm_loss, _nsp_loss, _mlm_acc, _nsp_acc = train.session.run(train.outfeed)
        mlm_loss = np.mean(_mlm_loss)
        nsp_loss = np.mean(_nsp_loss)
        mlm_acc = np.mean(_mlm_acc)
        nsp_acc = np.mean(_nsp_acc)
        if mlm_acc == -1 and nsp_acc == - 1:
            # If they are both disabled then it is worth to put Nan instead
            mlm_acc = np.nan
            nsp_acc = np.nan
    else:
        mlm_loss, nsp_loss = 0, 0
        mlm_acc, nsp_acc = 0, 0
    return batch_time, mlm_loss, nsp_loss, mlm_acc, nsp_acc


def train(opts):
    # --------------- OPTIONS ---------------------
    total_samples = dataset.get_dataset_files_count(opts, is_training=True)
    opts["dataset_repeat"] = math.ceil(
        (opts["num_train_steps"]*opts["global_batch_size"])/total_samples)

    total_samples_per_epoch = total_samples/opts["duplicate_factor"]
    logger.info(f"Total samples for each epoch {total_samples_per_epoch}")
    steps_per_epoch = total_samples_per_epoch//opts["global_batch_size"]
    logger.info(f"Total steps for each epoch {steps_per_epoch}")

    steps_per_logs = math.ceil(
        opts["steps_per_logs"] / opts['batches_per_step'])*opts['batches_per_step']
    steps_per_tensorboard = math.ceil(
        opts["steps_per_tensorboard"] / opts['batches_per_step'])*opts['batches_per_step']
    steps_per_ckpts = math.ceil(
        opts["steps_per_ckpts"] / opts['batches_per_step'])*opts['batches_per_step']
    logger.info(f"Checkpoint will be saved every {steps_per_ckpts} steps.")

    total_steps = (opts["num_train_steps"] //
                   opts['batches_per_step'])*opts['batches_per_step']
    logger.info(f"{opts['batches_per_step']} steps will be run for ipu to host synchronization once, it should be divided by num_train_steps, so num_train_steps will limit to {total_steps}.", opts)

    # learning rate strategy
    lr_schedule_name = opts['lr_schedule']
    logger.info(f"Using learning rate schedule {lr_schedule_name}")
    learning_rate_schedule = make_lr_schedule(lr_schedule_name, opts, total_steps)

    # variable loss scaling
    loss_scaling_schedule = LossScalingScheduler(opts['loss_scaling'], opts['loss_scaling_by_step'])

    # -------------- BUILD TRAINING GRAPH ----------------
    train = build_graph(opts,
                        is_training=True, feed_name="trainfeed")
    train.session.run(train.init)
    train.session.run(train.iterator.initializer)

    is_main_worker = opts['distributed_worker_index'] == 0

    step = 0
    # -------------- SAVE AND RESTORE --------------
    if opts["restore_dir"]:
        restore_path = opts['restore_dir']
        if os.path.isfile(restore_path):
            latest_checkpoint = os.path.splitext(restore_path)[0]
        else:
            latest_checkpoint = tf.train.latest_checkpoint(restore_path)
        logger.info(
            f"Restoring training from latest checkpoint: {latest_checkpoint}")
        step_pattern = re.compile(".*ckpt-([0-9]+)$")
        step = int(step_pattern.match(latest_checkpoint).groups()[0])
        train.saver.restore(train.session, latest_checkpoint)
        epoch = step / steps_per_epoch

        # restore event files
        source_path = os.path.join(opts["restore_dir"], '/event')
        target_path = os.path.join(opts["save_path"], '/event')
        if os.path.isdir(source_path):
            copytree(source_path, target_path)
    else:
        if opts["init_checkpoint"]:
            train.saver.restore(train.session, opts["init_checkpoint"])
            logger.info(
                f'Init Model from checkpoint {opts["init_checkpoint"]}')

    if opts['save_path']:
        file_path = train.saver.save(train.session, opts["checkpoint_path"], global_step=0)
        logger.info(f"Saved checkpoint to {file_path}")


    # Initialise Weights & Biases if available
    if opts['wandb'] and is_main_worker:
        import wandb
        wandb.init(project="tf-bert", sync_tensorboard=True)
        wandb.config.update(opts)

    # Tensorboard logs path
    log_path = os.path.join(opts["logs_path"], 'event')
    logger.info("Tensorboard event file path {}".format(log_path))
    summary_writer = tf.summary.FileWriter(
        log_path, train.graph, session=train.session)

    # ------------- TRAINING LOOP ----------------
    print_format = (
        "step: {step:6d}, epoch: {epoch:6.2f}, lr: {lr:6.7f}, mlm_loss: {mlm_loss:6.3f}, nsp_loss: {nsp_loss:6.3f},\
        mlm_acc: {mlm_acc:6.5f}, nsp_acc: {nsp_acc:6.5f}, samples/sec: {samples_per_sec:6.2f}, time: {iter_time:8.6f}, total_time: {total_time:8.1f}"
    )
    learning_rate = mlm_loss = nsp_loss = 0
    start_all = time.time()

    try:
        while step < total_steps:
            learning_rate = learning_rate_schedule.get_at_step(step)
            loss_scaling = loss_scaling_schedule.get_at_step(step)
            try:
                batch_time, mlm_loss, nsp_loss, mlm_acc, nsp_acc = training_step(
                    train, learning_rate, loss_scaling)
            except tf.errors.OpError as e:
                raise tf.errors.ResourceExhaustedError(
                    e.node_def, e.op, e.message)

            batch_time /= opts['batches_per_step']

            is_log_step = (step % steps_per_logs == 0)
            is_save_tensorboard_step = (steps_per_tensorboard != 0 and (
                step % steps_per_tensorboard == 0))
            is_save_ckpt_step = (step and (
                step % steps_per_ckpts == 0 or step == total_steps - opts['batches_per_step']))

            if (step == 1 and (is_main_worker or opts['log_all_workers'])):
                poplar_compile_time = time.time() - start_all
                logger.info(f"Poplar compile time: {poplar_compile_time:.2f}s")
                poplar_summary = tf.Summary()
                poplar_summary.value.add(
                    tag='poplar/compile_time', simple_value=poplar_compile_time)
                summary_writer.add_summary(poplar_summary)

            if is_log_step:
                total_time = time.time() - start_all
                epoch = step / steps_per_epoch
                stats = OrderedDict([
                    ('step', step),
                    ('epoch', epoch),
                    ('lr', learning_rate),
                    ('loss_scaling', loss_scaling),
                    ('mlm_loss', mlm_loss),
                    ('nsp_loss', nsp_loss),
                    ('mlm_acc', mlm_acc),
                    ('nsp_acc', nsp_acc),
                    ('iter_time', batch_time),
                    ('samples_per_sec', opts['global_batch_size']/batch_time),
                    ('total_time', total_time),
                ])

                logger.info(print_format.format(**stats))

            # Log training statistics
            train_summary = tf.Summary()
            train_summary.value.add(tag='epoch', simple_value=epoch)
            train_summary.value.add(tag='loss/MLM', simple_value=mlm_loss)
            train_summary.value.add(tag='loss/NSP', simple_value=nsp_loss)
            train_summary.value.add(tag='accuracy/MLM', simple_value=mlm_acc)
            train_summary.value.add(tag='accuracy/NSP', simple_value=nsp_acc)
            train_summary.value.add(
                tag='learning_rate', simple_value=learning_rate)
            train_summary.value.add(
                tag='loss_scaling', simple_value=loss_scaling)
            train_summary.value.add(
                tag='samples_per_sec', simple_value=opts['global_batch_size']/batch_time)
            train_summary.value.add(
                tag='samples', simple_value=step*opts['batches_per_step']*opts['global_batch_size'])
            summary_writer.add_summary(train_summary, step)
            summary_writer.flush()

            if is_save_ckpt_step or is_save_tensorboard_step:
                if is_main_worker:
                    file_path = train.saver.save(train.session, opts["checkpoint_path"], global_step=step)
                    logger.info(f"Saved checkpoint to {file_path}")

                    if is_save_tensorboard_step:
                        log.save_model_statistics(file_path, summary_writer, step)

                if opts['use_popdist']:
                    ipu_utils.barrier()

            step += opts['batches_per_step']
    finally:
        train.session.close()


def str_to_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise argparse.ArgumentTypeError(f'{value} is not a valid boolean value')


def add_main_arguments(parser):
    group = parser.add_argument_group('Main')
    group.add_argument('--help', action='store_true', default=False,
                       help="Display help.")
    group.add_argument('--task', type=str, choices=['pretraining'],
                       help="Type of NLP task.")
    group.add_argument('--config', type=str,
                       help='BERT configuration file in JSON format.')
    return parser


def add_other_arguments(parser, required=True):
    group = parser.add_argument_group('Main')

    # Training options
    group.add_argument('--batch-size', type=int,
                       help="Set batch-size for training graph")
    group.add_argument('--global-batch-size', type = int, default = None,
                       help="The total batch size at which we want the model to run")
    group.add_argument('--base-learning-rate', type=float, default=2e-5,
                       help="Base learning rate exponent (2**N). blr = lr /  bs")
    group.add_argument('--num-train-steps', type=int,
                       help="Number of training steps.")
    group.add_argument('--loss-scaling', type=float, default=1,
                       help="Loss scaling factor.")
    group.add_argument('--loss-scaling-by-step', type=str, default=None,
                       help="Specify changing loss scaling factors at given training steps, as a dictionary.")
    group.add_argument('--steps-per-ckpts', type=int, default=256,
                       help="Steps per checkpoints")
    group.add_argument('--optimizer', type=str, default="momentum",
                       choices=['sgd', 'momentum', 'adamw', 'lamb'],
                       help="Optimizer")
    group.add_argument('--momentum', type=float, default=0.984375,
                       help="Momentum coefficient.")
    group.add_argument('--beta1', type=float, default=0.9,
                       help="lamb/adam beta1 coefficient.")
    group.add_argument('--beta2', type=float, default=0.999,
                       help="lamb/adam beta2 coefficient.")
    group.add_argument('--weight-decay-rate', type=float, default=0.0,
                       help="Weight decay to use during optimisation.")
    group.add_argument('--epsilon', type=float,
                       default=1e-4, help="lamb/adam epsilon.")
    group.add_argument('--lr-schedule', default='exponential',
                       choices=["custom", "natural_exponential", "polynomial"],
                       help="Learning rate schedule function.")
    group.add_argument('--lr-schedule-by-step', type=str,
                       help="Dictonary of changes in the learning rate at specified steps.")
    group.add_argument('--warmup', default=0.1,
                       help="Learning rate schedule warm-up period, in epochs (float) or number of steps (integer).")
    group.add_argument('--seed', default=None,
                       help="Seed for randomizing training")
    group.add_argument('--wandb', action='store_true',
                       help="Enable logging and experiment tracking with Weights & Biases.")
    group.add_argument('--save-path', type=str, default="checkpoints",
                       help='Save checkpoints to this directory.')
    group.add_argument('--init-checkpoint', type=str, default=None,
                       help='Initialise a new training session from this checkpoint.')
    group.add_argument('--restore-dir', type=str, default=None,
                       help='Path to directory containing the checkpoint to restore.')
    group.add_argument('--restore-optimiser-from-checkpoint', default=True, action="store_true")
    group.add_argument('--save-optimiser-to-checkpoint', default=True, action="store_true")
    group.add_argument('--disable-acc', default=False, action='store_true',
                       help='If passed, this flag disables the calculation of the accuracies to save some memory on the first IPU of the pipeline.')

    # BERT options
    group.add_argument('--vocab-size', type=int,
                       help="""Vocabulary size of `inputs_ids` in `BertModel`.""")
    group.add_argument('--hidden-size', type=int,
                       help="""Size ofthe encoder layers and the pooler layer.""")
    group.add_argument('--num-hidden-layers', type=int,
                       help="""Number of hidden layers in the Transformer encoder.""")
    group.add_argument('--num-attention-heads', type=int,
                       help="""Number of attention heads for each attention layer in the Transformer encoder.""")
    group.add_argument('--intermediate-size', type=int,
                       help="""The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.""")
    group.add_argument('--hidden-act', type=int,
                       help="""The non-linear activation function (function or string) in the encoder and pooler.""")
    group.add_argument('--hidden-dropout-prob', type=int,
                       help="""The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.""")
    group.add_argument('--attention-probs-dropout-prob', type=int,
                       help="""The dropout ratio for the attention probabilities.""")
    group.add_argument('--max-position-embeddings', type=int,
                       help= """The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).""")
    group.add_argument('--type-vocab-size', type=int,
                       help= """The vocabulary size of the `token_type_ids` passed into `BertModel`.""")
    group.add_argument('--initializer-range', type=int,
                       help= """The stdev of the truncated-normal-initializer for initializing all weight matrices.""")

    # Model options
    group.add_argument('--use-attention-projection-bias', type=str_to_bool, default=True,
                       help="Whether to use bias in linear projection behind attention layer.")
    group.add_argument('--use-cls-layer', type=str_to_bool, default=True,
                       help="""Include the CLS layer in pretraining.
                       This layer comes after the encoders but before the projection for the MLM loss.""")
    group.add_argument('--use-qkv-bias', type=str_to_bool, default=True,
                       help="""Whether to use bias in QKV calculation of attention layer.""")

    # IPU options
    pipeline_schedule_options = [
        _.name for _ in ipu.ops.pipelining_ops.PipelineSchedule]
    schedulers_available = ['Clustering',
                            'PostOrder', 'LookAhead', 'ShortestPath']
    recomputation_modes_available = [
        p.name for p in ipu.ops.pipelining_ops.RecomputationMode
    ]

    group.add_argument('--gradient-accumulation-count', type=int, default=None,
                       help="Number of gradients to accumulate in the pipeline. Must also set --shards > 1.")
    group.add_argument('--pipeline-schedule', type=str, default="Interleaved",
                       choices=pipeline_schedule_options, help="Pipelining scheduler.")
    group.add_argument('--replicas', type=int, default=1,
                       help="Replicate graph over N workers to increase batch to batch-size*N")
    group.add_argument('--precision', type=str, default="16", choices=["16", "32"],
                       help="Precision of Ops(weights/activations/gradients) data types: 16, 32.")
    group.add_argument('--batches-per-step', type=int, default=1,
                       help="Maximum number of batches to perform on the device before returning to the host.")
    group.add_argument('--available-memory-proportion', type=str, default=0.23,
                       help="Proportion of IPU memory available to matmul operations. A list can be used to specify the value for each IPU.")
    group.add_argument('--variable-offloading', type=str_to_bool, default=True,
                       help="Enable offloading of training variables into remote memory.")
    group.add_argument('--stochastic-rounding', type=str_to_bool, default=True,
                       help="Enable stochastic rounding. Set to False when run evaluation.")
    group.add_argument('--no-outlining', type=str_to_bool, default=False,
                       help="Disable TF outlining optimisations. This will increase memory for a small throughput improvement.")
    group.add_argument("--xla_recompute", default=True, action="store_true",
                       help="Recompute activations during backward pass")
    group.add_argument('--fp-exceptions', default=False, action="store_true",
                       help="Enable floating-point exeptions.")
    group.add_argument('--partials-type', type=str, default="half", choices=["half", "float"],
                       help="Mamul&Conv precision data type.")
    group.add_argument('--max-cross-replica-sum-buffer-size', type=int, default=10*1024*1024,
                       help="""The maximum number of bytes that can be waiting before a cross replica sum op is scheduled. [Default=10*1024*1024]""")
    group.add_argument('--scheduler', type=str, default='Clustering', choices=schedulers_available,
                       help="""Forces the compiler to use a specific scheduler when ordering the instructions.""")
    group.add_argument('--recomputation-mode', type=str, default="RecomputeAndBackpropagateInterleaved",
                       choices=recomputation_modes_available)
    group.add_argument('--increase-optimiser-precision', action='store_true', default=False,
                       help='In the LAMB optimiser, it performs more operations in fp32. This operation increase precision in the weight update but consumes more memory and reduce the Tput.')
    group.add_argument('--use-nvlamb', action='store_true', default=False,
                       help="Flag to use the global normalisation for the gradients.")
    group.add_argument('--use-debiasing', action='store_true', default=False,
                       help="Flag to use the de biasing for the momenta of LAMB")
    group.add_argument('--duplicate-factor', default=5, type=int,
                       help='The amount of duplication factor inside the dataset.')
    group.add_argument('--reduction-type', type=str, choices=['sum', 'mean'], default='mean',
                       help='The reduction type applied to the pipeline, the choice is between summation and mean.')
    group.add_argument('--weight-norm-clip', type=float, default=0.,
                       help='The value from which we want to clip the w_norm value, value of 0 is no weight clipping.')
    group.add_argument('--compile-only', action="store_true", default=False,
                       help="Configure Poplar to only compile the graph. This will not acquire any IPUs and thus facilitate profiling without using hardware resources.")
    group.add_argument('--matmul-serialize-factor', type=int, default=6,
                       help='Serialization factor of the embeddings lookup and projection. Must be a divisor of vocab_size.')
    group.add_argument('--use-qkv-split', action='store_true', default=False,
                       help='split the QKV layer into independent tensors.')
    group.add_argument('--pipeline-stages', type=str,
                       help="""Pipeline stages, a list of [emb, pos, hid, mlm, nsp] layers forming the pipeline.""")
    group.add_argument('--device-mapping', type=str,
                       help="""Mapping of pipeline stages to IPU""")
    group.add_argument('--sync-replicas-independently', action='store_true', default=False,
                       help='All the replicas will be in sync.')
    group.add_argument('--log-all-workers', action='store_true',
                       help='Allow all the workers to log into the terminal and the files.')

    # Dataset options
    group.add_argument('--train-file', type=str, required=False,
                       help="path to wiki/corpus training dataset tfrecord file.")
    group.add_argument("--seq-length", type=int, default=128,
                       help="the max sequence length.")
    group.add_argument("--max-predictions-per-seq", type=int, default=20,
                       help="the number of masked words per sentence.")
    group.add_argument('--parallell-io-threads', type=int, default=4,
                       help="Number of cpu threads used to do data prefetch.")
    group.add_argument('--generated-data', action="store_true", default=False,
                       help="Generates synthetic-data on the host and then use it for training.")
    group.add_argument('--synthetic-data', action='store_true',
                       help="Run the model completely detaching it from the host.")
    group.add_argument('--dataset-repeat', type=int, default=1,
                       help="Number of times dataset to repeat.")
    group.add_argument('--static-mask', action='store_true', default=False,
                       help="Use if the pretraining dataset was created with the masked tokens always at the beginning of the input tensor.")

    # Env flag specific arguments
    group.add_argument('--execution-profile', action='store_true',
                       help='Sets the Poplar engine options to output an execution profile to the profile-dir.')
    group.add_argument('--memory-profile', action='store_true',
                       help='Sets the Poplar engine options to output a memory profile to the profile-dir.')
    group.add_argument('--profile-dir', type=str, default='./',
                       help='Defines the directory where the profile will be found.')
    group.add_argument('--progress-bar', type=str, choices=['auto', 'true', 'false'], default='auto',
                       help='The compilation progress bar for the compilation. Pass false to disable it.')

    # Add logging-specific arguments
    log.add_arguments(parser)

    return parser


def create_command_line_parser():
    parser = argparse.ArgumentParser(
        description='BERT  Pretraining in TensorFlow',
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_main_arguments(parser)

    return parser


def set_distribution_defaults(opts):

    if opts['use_popdist']:
        opts['distributed_worker_count'] = popdist.getNumInstances()
        opts['distributed_worker_index'] = popdist.getInstanceIndex()
        opts['summary_str'] += 'Popdist\n'
        opts['summary_str'] += ' Process count: {distributed_worker_count}\n'
        opts['summary_str'] += ' Process index: {distributed_worker_index}\n'
    else:
        opts['distributed_worker_count'] = 1
        opts['distributed_worker_index'] = 0

    if opts['distributed_worker_index'] != 0 and not opts['log_all_workers']:
        logger.setLevel(logging.ERROR)


def create_all_options_parser():
    parser = argparse.ArgumentParser(
        description='BERT  Pretraining in TensorFlow',
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_main_arguments(parser)
    parser = add_other_arguments(parser)
    return parser


def set_training_defaults(opts):
    opts['name'] = 'BERT_' + opts['task']
    opts['summary_str'] = "Training\n"
    opts['summary_str'] += " Batch Size: {global_batch_size}\n"
    opts['summary_str'] += (" Base Learning Rate: {base_learning_rate}\n"
                            " Loss Scaling: {loss_scaling}\n")

    if opts['optimizer'].lower() == 'sgd':
        opts['summary_str'] += "SGD\n"
    elif opts['optimizer'].lower() == 'momentum':
        opts['name'] += '_Mom'
        opts['summary_str'] += ("SGD with Momentum\n"
                                " Momentum Coefficient: {momentum}\n")
    elif opts['optimizer'].lower() == 'adam':
        opts['name'] += '_Adam'
        opts['summary_str'] += ("Adam\n"
                                " beta1: {beta1}, beta2: {beta2}, epsilon: {epsilon}\n")
    elif opts['optimizer'].lower() == 'adamw':
        opts['name'] += '_AdamW'
        opts['summary_str'] += ("Adam With Weight decay\n"
                                " beta1: {beta1}, beta2: {beta2}, epsilon: {epsilon}\n")
    elif opts['optimizer'].lower() == 'lamb':
        opts['name'] += '_LAMB'
        opts['summary_str'] += ("LAMB\n"
                                " beta1: {beta1}, beta2: {beta2}, epsilon: {epsilon}\n")

    # Automatic pipeline depth counter
    if opts["global_batch_size"]:
        gradients_to_accumulate = opts["global_batch_size"]//(opts["total_replicas"]*opts['batch_size'])
        divisor = len(opts['pipeline_stages'])*2
        # We need then to fix the gradient_to_accumulate according to the pipeline
        gradients_to_accumulate = divisor*(1 + gradients_to_accumulate//divisor)
        if opts['gradient_accumulation_count'] and opts['gradient_accumulation_count'] != gradients_to_accumulate:
            logger.error("Passed a gradient to accumulate and a global batch size. Disable one of them to run.")
            sys.exit(os.EX_OK)
        opts['gradient_accumulation_count'] = gradients_to_accumulate
        # We update the global_batch_size
        proposed_global_batch_size = opts['gradient_accumulation_count'] * opts["total_replicas"] * opts["batch_size"]
        if proposed_global_batch_size != opts['global_batch_size']:
            logger.info("Changing the global batch size to match the pipeline requirements.")
            opts['global_batch_size'] = proposed_global_batch_size
    else:
        opts['global_batch_size'] = opts['batch_size'] * opts['gradient_accumulation_count']*opts['total_replicas']

    if opts['gradient_accumulation_count'] > 1:
        opts['summary_str'] += "  Gradient Accumulation Count {gradient_accumulation_count} \n"

    # In order to improve readability we set another flag
    opts['compute_acc'] = not opts['disable_acc']
    if opts['disable_acc']:
        logger.info("Disabling computation of the accuracies. Just the losses will be reported.")


def set_ipu_defaults(opts):
    poplar_version = os.popen('popc --version').read()
    opts['poplar_version'] = poplar_version
    logger.info(f"Running on host: {gethostname()}")
    logger.info(f"Current date/time: {str(datetime.datetime.now())}")
    commit_hash = log.get_git_revision()
    logger.info(f"Code revision: {commit_hash}")

    if opts['seed']:
        # Seed the various random sources
        seed = opts['seed']
        logger.info(f"Pseudo-random number generator seed specified: f{seed}")
        random.seed(seed)
        # Set other seeds to different values for extra safety
        tf.set_random_seed(random.randint(0, 2**32 - 1))
        np.random.seed(random.randint(0, 2**32 - 1))
        ipu.utils.reset_ipu_seed(random.randint(-2**16, 2**16 - 1))


def set_defaults(opts):
    opts['summary_str'] = "\n"
    dataset.set_defaults(opts)
    set_distribution_defaults(opts)
    set_training_defaults(opts)
    set_ipu_defaults(opts)
    log.set_defaults(opts)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Parse command-line arguments
    command_line_parser = create_command_line_parser()
    all_options_parser = create_all_options_parser()

    known_command_line_args, unknown_command_line_args = command_line_parser.parse_known_args()

    if known_command_line_args.help or known_command_line_args.config is None:
        all_options_parser.print_help()
        sys.exit(os.EX_OK)

    # Parse options specified in the configuration file into
    config_file_path = known_command_line_args.config
    opts_from_config_file = bert_ipu.BertConfig.from_json_file(config_file_path)

    # Build the global options structure from the default options
    current_options = vars(all_options_parser.parse_args())

    unknown_options = [
        opt for opt in opts_from_config_file.keys()
        if opt not in current_options.keys()
    ]

    if unknown_options:
        logger.error(f"Unonwn options: {unknown_options}")
        sys.exit(os.EX_USAGE)

    # Overwrite global options by those specified in the config file.
    current_options.update(opts_from_config_file)
    options_namespace = argparse.Namespace(**current_options)

    # Overwrite with command-line arguments
    all_options_namespace = all_options_parser.parse_args(unknown_command_line_args, options_namespace)

    # argparse.Namespace -> dict()
    opts = vars(all_options_namespace)

    opts['shards'] = ipu_utils.next_power_of_two(max(opts["device_mapping"]) + 1)

    if popdist.isPopdistEnvSet():
        opts['use_popdist'] = True
        opts['replicas'] = popdist.getNumLocalReplicas()
        opts['total_replicas'] = popdist.getNumTotalReplicas()
        if opts['compile_only']:
            opts['select_ipu'] = None
        else:
            opts['select_ipu'] = popdist.getDeviceId()
    else:
        opts['use_popdist'] = False
        opts['total_replicas'] = opts['replicas']
        opts['select_ipu'] = None

    set_defaults(opts)

    set_poplar_engine_options(
        execution_profile=opts['execution_profile'],
        memory_profile=opts['memory_profile'],
        profile_dir=str(opts['profile_dir']),
        sync_replicas_independently=opts['replicas'] > 1 and opts['sync_replicas_independently'],
        synthetic_data=opts['synthetic_data'],
        tensorflow_progress_bar=opts['progress_bar']
    )

    logger.info(f"Overwrite configuration parameters: {', '.join(unknown_command_line_args)}")

    if unknown_options:
        logger.error(f"Unonwn options: {unknown_options}")
        sys.exit(os.EX_USAGE)

    poplar_options = os.getenv('POPLAR_ENGINE_OPTIONS', 'unset')
    logger.info(f"Poplar options: {poplar_options}")
    logger.info("Command line: " + ' '.join(sys.argv))
    if opts['use_popdist'] and opts['log_all_workers']:
        option_string = f"Option flags for worker {opts['distributed_worker_index']}:\n"
    else:
        option_string = f"Option flags:\n"
    logger.info(option_string + json.dumps(
            OrderedDict(sorted(opts.items())), indent=1))

    # Start training
    train(opts)
