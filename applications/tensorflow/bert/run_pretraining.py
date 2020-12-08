#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
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

import argparse
import datetime
import json
import os
import random
import re
import sys
import time
from collections import OrderedDict, deque, namedtuple
from functools import partial
from socket import gethostname

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu

import Datasets.data as dataset
import log as bert_logging
import Models.modeling_ipu as bert_ipu
from lr_schedules import make_lr_schedule
import optimization
from ipu_utils import get_ipu_config, ladder_numbering_iterator, next_power_of_two
from log import logger

# Data structure containing the network state
GraphOps = namedtuple(
    'graphOps', ['graph',
                 'session',
                 'init',
                 'ops',
                 'placeholders',
                 'iterator',
                 'outfeed',
                 'saver',
                 'restore',
                 'tvars'])


def get_output_stage(learning_rate, layer_output, pooled_output,
                     masked_labels, next_sentence_labels, masked_lm_weights,
                     bert_model, matmul_serialization_factor):
    """Calculate MLM and NSP losses"""

    data_type = bert_model.bert_config.dtype

    mlm_logits = bert_model.mlm_head(layer_output)

    # Calculate MLM loss
    with tf.variable_scope("cls/predictions"):
        log_probs = tf.nn.log_softmax(mlm_logits, axis=-1)
        label_ids = tf.reshape(masked_labels, [-1])
        label_weights = tf.reshape(masked_lm_weights, [-1])

        one_hot_labels = tf.one_hot(tf.cast(label_ids, dtype=tf.int32),
                                    depth=bert_model.bert_config.vocab_size, dtype=data_type)

        log_probs_per_example = tf.reshape(log_probs, [log_probs.shape[0]*log_probs.shape[1], log_probs.shape[2]])
        per_example_loss = -tf.reduce_sum(log_probs_per_example * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        mlm_loss = numerator / denominator
        mlm_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.cast(tf.argmax(log_probs_per_example, -1), dtype=tf.int32), label_ids), dtype=data_type))

    # Calculate NSP loss
    nsp_logits = bert_model.nsp_head(pooled_output)
    with tf.variable_scope("cls/seq_relationship"):
        # Google used loss
        log_probs = tf.nn.log_softmax(nsp_logits, axis=-1)
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
        one_hot_labels = tf.one_hot(next_sentence_labels, depth=2, dtype=data_type)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        nsp_loss = tf.reduce_mean(per_example_loss)
        nsp_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.cast(tf.argmax(log_probs, -1), dtype=tf.int32), next_sentence_labels), dtype=data_type))

    return learning_rate, mlm_loss, nsp_loss, mlm_acc, nsp_acc


def basic_pipelined_training_step(infeed,
                                  outfeed,
                                  iterations_per_step=1,
                                  bert_config=None,
                                  opts=None,
                                  learning_rate=None,
                                  is_training=True):
    """ Re-create BERT model as a pipeline of individual stages """
    model = bert_ipu.BertModel(bert_config, is_training=is_training)
    # The computational stages composing the pipeline
    computational_stages = []
    device_mapping = []
    device_counter = iter(ladder_numbering_iterator())

    # These variables will be set depending on the embeddings split across IPUs
    word_embedding_device = -1
    positional_embedding_device = -1

    if opts['embeddings_placement'].lower() == 'two_ipus':
        computational_stages.append(model.embedding_lookup_stage)
        device_mapping.append(next(device_counter))
        word_embedding_device = device_mapping[-1]

        computational_stages.append(model.embedding_postprocessor_stage)
        device_mapping.append(next(device_counter))
        positional_embedding_device = device_mapping[-1]

    elif opts['embeddings_placement'].lower() == 'same_ipu':
        def word_and_positional_embeddings_same_stage(*inputs):
            word_embedding_outputs = model.embedding_lookup_stage(*inputs)
            positional_embedding_output = model.embedding_postprocessor_stage(
                *word_embedding_outputs)
            return positional_embedding_output

        computational_stages.append(word_and_positional_embeddings_same_stage)
        device_mapping.append(next(device_counter))
        word_embedding_device = device_mapping[-1]
        positional_embedding_device = word_embedding_device

    elif opts['embeddings_placement'].lower() == 'same_as_hidden_layers':
        # We will add the embeddings later, when we create the frst stage of hidden layers
        pass

    else:
        raise ValueError("Configuration parameter 'embeddings_placement' not recognised.")

    # Place a multiple hidden layers into a same stage.
    first_layer_idx = 0
    for i, num_hidden_layer_in_stage in enumerate(bert_config.hidden_layers_per_stage):
        logger.info(f"Adding {num_hidden_layer_in_stage} hidden layers in stage {i}")

        # Place the embeddings at the same time as the first few layers if
        # the option 'embeddings_placement' is set to 'same_as_hidden_layers'
        if i == 0 and opts['embeddings_placement'].lower() == 'same_as_hidden_layers':
            # Put all embeddings and the first hidden layers in the same stage.
            computational_stages.append(partial(
                model.embeddings_and_hidden_layers_stage, first_layer_idx, num_hidden_layer_in_stage))
            device_mapping.append(next(device_counter))
            word_embedding_device = device_mapping[-1]
            positional_embedding_device = device_mapping[-1]
        else:
            # Put multiple hidden layers in the same stage
            computational_stages.append(partial(
                model.multi_hidden_layers_stage, first_layer_idx, num_hidden_layer_in_stage))
            device_mapping.append(next(device_counter))

        first_layer_idx += num_hidden_layer_in_stage

    # We check that the two embeddings have been assigned to an IPU
    assert word_embedding_device >= 0
    assert positional_embedding_device >= 0

    # Place the loss functions on the first IPU
    computational_stages.append(partial(
        get_output_stage,
        bert_model=model, matmul_serialization_factor=opts['logits_matmul_serialization_factor']))
    device_mapping.append(word_embedding_device)

    logger.info("*** Pipeline summary ***")
    for stage, device in zip(computational_stages, device_mapping):
        logger.info(f"IPU {device:2d} - stage {str(stage):50s}")

    def optimizer_function(learning_rate, mlm_loss, nsp_loss, mlm_acc, nsp_acc):
        total_loss = mlm_loss + nsp_loss
        # Choose optimiser method from configuration flags.
        optimizer = optimization.get_optimizer(learning_rate, opts)

        if opts["replicas"] > 1:
            optimizer = ipu.optimizers.cross_replica_optimizer.CrossReplicaOptimizer(
                optimizer)
        return ipu.ops.pipelining_ops.OptimizerFunctionOutput(optimizer, total_loss*opts["loss_scaling"])

    options = [ipu.pipelining_ops.PipelineStageOptions()] * len(device_mapping)

    if is_training:
        pipeline_ops = ipu.ops.pipelining_ops.pipeline(computational_stages=computational_stages,
                                                       gradient_accumulation_count=int(
                                                           opts['pipeline_depth']),
                                                       repeat_count=iterations_per_step,
                                                       inputs=[learning_rate],
                                                       infeed_queue=infeed,
                                                       outfeed_queue=outfeed,
                                                       optimizer_function=optimizer_function,
                                                       device_mapping=device_mapping,
                                                       forward_propagation_stages_poplar_options=options,
                                                       backward_propagation_stages_poplar_options=options,
                                                       offload_weight_update_variables=opts[
                                                           'offload_weight_update_variables'],
                                                       pipeline_schedule=ipu.ops.pipelining_ops.PipelineSchedule[opts['pipeline_schedule']],
                                                       recomputation_mode=ipu.ops.pipelining_ops.RecomputationMode[opts['recomputation_mode']],
                                                       name="Pipeline")
    else:
        pipeline_ops = ipu.ops.pipelining_ops.pipeline(computational_stages=computational_stages,
                                                       gradient_accumulation_count=int(
                                                           opts['pipeline_depth']),
                                                       repeat_count=iterations_per_step,
                                                       inputs=[learning_rate],
                                                       infeed_queue=infeed,
                                                       outfeed_queue=outfeed,
                                                       device_mapping=device_mapping,
                                                       forward_propagation_stages_poplar_options=options,
                                                       backward_propagation_stages_poplar_options=options,
                                                       offload_weight_update_variables=opts[
                                                           'offload_weight_update_variables'],
                                                       pipeline_schedule=opts['pipeline_schedule'],
                                                       recomputation_mode=opts['recomputation_mode'],
                                                       name="Pipeline")

    return pipeline_ops


def training_step_with_infeeds_and_outfeeds(bert_config, train_iterator, outfeed, opts, learning_rate, iterations_per_step=1, is_training=True):

    if int(opts['pipeline_depth']) > 1:

        _training_step = partial(basic_pipelined_training_step,
                                 infeed=train_iterator,
                                 outfeed=outfeed,
                                 iterations_per_step=iterations_per_step,
                                 bert_config=bert_config,
                                 opts=opts,
                                 learning_rate=learning_rate)

        return ipu.ipu_compiler.compile(_training_step, [])
    else:
        raise NotImplementedError("Single-stage pipeline not supported by BERT pretraining.")


def build_graph(bert_config, opts, iterations_per_step=1, is_training=True, feed_name=None):
    """Build the graph for training.

    Args:
        bert_config: configuration for the BERT model.
        opts: a dictionary containing all global options.
        iterations_per_step: number of iterations per step
        is_training (bool): if true return a graph with trainable variables.
        feed_name: name of the IPU infeed.

    Returns:
        a GraphOps containing a BERT graph and session prepared for inference or training.
    """
    train_graph = tf.Graph()
    with train_graph.as_default():

        placeholders = dict()
        placeholders['learning_rate'] = tf.placeholder(
            bert_config.dtype, shape=[])
        learning_rate = placeholders['learning_rate']

        train_iterator = ipu.ipu_infeed_queue.IPUInfeedQueue(
            dataset.data(opts, is_training=is_training), feed_name=feed_name+"_in", replication_factor=opts['replicas'])

        outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
            feed_name=feed_name+"_out", replication_factor=opts['replicas'])

        with ipu.scopes.ipu_scope('/device:IPU:0'):
            train = training_step_with_infeeds_and_outfeeds(bert_config, train_iterator, outfeed_queue,
                                                            opts, learning_rate, iterations_per_step, is_training=is_training)

        outfeed = outfeed_queue.dequeue()

        bert_logging.print_trainable_variables(opts['logs_path'])

        model_variables = tf.trainable_variables() + tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)
        model_and_optimiser_variables = tf.global_variables()

        restore = tf.train.Saver(var_list=model_and_optimiser_variables if opts['restore_optimiser_from_ckpt'] else model_variables)

        # We store two savers: one for the standard training and another one for the best checkpoint
        savers = {
            "train_saver": tf.train.Saver(var_list=model_variables if opts['ckpt_model_only'] else model_and_optimiser_variables, name='latest', max_to_keep=5),
            "best_saver": tf.train.Saver(var_list=model_variables if opts['ckpt_model_only'] else model_and_optimiser_variables, name='best', max_to_keep=1)
        }

        ipu.utils.move_variable_initialization_to_cpu()
        train_init = tf.global_variables_initializer()
        tvars = tf.trainable_variables()

    # Calculate number of IPUs required for pretraining pipeline.
    num_embedding_ipu = {
        'two_ipus': 2,
        'same_ipu': 1,
        'same_as_hidden_layers': 0
    }[opts['embeddings_placement']]

    num_hidden_layer_stages = len(bert_config.hidden_layers_per_stage)
    num_ipus_required = opts['replicas'] * next_power_of_two(num_hidden_layer_stages + num_embedding_ipu)

    # Configure the IPU options.
    ipu_options = get_ipu_config(
        fp_exceptions=opts["fp_exceptions"],
        stochastic_rounding=opts['stochastic_rounding'],
        xla_recompute=opts["xla_recompute"],
        available_memory_proportion=opts['available_memory_proportion'],
        disable_graph_outlining=opts["disable_graph_outlining"],
        num_ipus_required=num_ipus_required,
        max_cross_replica_sum_buffer_size=opts['max_cross_replica_sum_buffer_size'],
        scheduler_selection=opts['scheduler'],
        compile_only=opts['compile_only'],
        partials_type=opts['partials_type']
    )
    ipu.utils.configure_ipu_system(ipu_options)

    train_sess = tf.Session(graph=train_graph, config=tf.ConfigProto())

    return GraphOps(train_graph, train_sess, train_init, [train], placeholders, train_iterator, outfeed, savers, restore, tvars)


def training_step(train, learning_rate):
    """Run a single training step, over a global batch.
    Args:
        train (GraphOps): GraphOps containing the current training session and graph.
        learning_rate: learnign at rate this training step.
    """
    # Run Training
    start = time.time()
    _ = train.session.run(train.ops, feed_dict={
        train.placeholders['learning_rate']: learning_rate,
    })
    batch_time = (time.time() - start)
    if not os.environ.get('TF_POPLAR_FLAGS') or '--use_synthetic_data' not in os.environ.get('TF_POPLAR_FLAGS'):
        _learning_rate, _mlm_loss, _nsp_loss, _mlm_acc, _nsp_acc = train.session.run(train.outfeed)
        mlm_loss = np.mean(_mlm_loss)
        nsp_loss = np.mean(_nsp_loss)
        mlm_acc = np.mean(_mlm_acc)
        nsp_acc = np.mean(_nsp_acc)
    else:
        mlm_loss, nsp_loss, mlm_acc, nsp_acc = 0, 0, 0, 0
    return batch_time, mlm_loss, nsp_loss, mlm_acc, nsp_acc


def train(bert_config, opts):
    # --------------- OPTIONS ---------------------
    epochs = opts["epochs"]
    total_samples = dataset.get_dataset_files_count(opts, is_training=True)
    logger.info("Total samples with duplications {}".format(total_samples))
    total_independent_samples = total_samples//opts['duplication_factor']
    logger.info("Total samples without duplications {}".format(total_independent_samples))
    steps_per_epoch = total_independent_samples // (opts['batches_per_step']*opts["total_batch_size"])
    iterations_per_epoch = total_independent_samples // (opts["total_batch_size"])

    # total iterations
    if opts['steps']:
        logger.warn("Ignoring the epoch flag and using the steps one")
        steps = opts['steps']
    else:
        steps = epochs * steps_per_epoch
    logger.info("Total training steps equal to {}, total number of samples being analyzed equal to {}".format(steps, steps*opts['batches_per_step']*opts['total_batch_size']))
    iterations_per_step = opts['batches_per_step']
    ckpt_per_step = opts['steps_per_ckpts']

    # avoid nan issue caused by queue length is zero.
    queue_len = iterations_per_epoch // iterations_per_step
    if queue_len == 0:
        queue_len = 1
    batch_times = deque(maxlen=queue_len)

    # learning rate strategy
    lr_schedule_name = opts['lr_schedule']
    logger.info(f"Using learning rate schedule {lr_schedule_name}")
    LR = make_lr_schedule(lr_schedule_name, opts, steps)

    if opts['do_train']:
        # -------------- BUILD TRAINING GRAPH ----------------

        train = build_graph(bert_config, opts, iterations_per_step, is_training=True, feed_name="trainfeed")
        train.session.run(train.init)
        train.session.run(train.iterator.initializer)

        step = 0
        i = 0

        if opts['restore_path'] is not None:
            if os.path.isdir(opts['restore_path']):
                ckpt_file_path = tf.train.latest_checkpoint(opts['restore_path'])
                logger.info(f"Restoring training from latest checkpoint")
            else:
                # Assume it's a directory
                ckpt_file_path = opts['restore_path']

            logger.info(f"Restoring training from checkpoint: {ckpt_file_path}")
            train.restore.restore(train.session, ckpt_file_path)

            ckpt_pattern = re.compile(".*ckpt-([0-9]+)$")
            i = int(ckpt_pattern.match(ckpt_file_path).groups()[0])
            step = int(i//iterations_per_step)

        if opts['start_from_ckpt']:
            # We use a checkpoint to initialise our model
            train.restore.restore(train.session, opts['start_from_ckpt'])
            logger.info("Starting the training from the checkpoint {}".format(opts['start_from_ckpt']))

        # Initialise Weights & Biases if available
        if opts['wandb']:
            import wandb
            wandb.init(project="tf-bert", sync_tensorboard=True)
            wandb.config.update(opts)

        # Tensorboard logs path
        log_path = os.path.join(opts["logs_path"], 'event')
        logger.info("Tensorboard event file path {}".format(log_path))
        summary_writer = tf.summary.FileWriter(log_path, train.graph, session=train.session)

        # ------------- TRAINING LOOP ----------------
        logger.info(
            "################################################################################")
        logger.info("Start training......")
        print_format = (
            "step: {step:6d}, iteration: {iteration:6d}, epoch: {epoch:6.3f}, lr: {lr:10.3g}, mlm_loss: {mlm_loss:6.3f}, nsp_loss: {nsp_loss:6.3f}, "
            "samples/sec: {samples_per_sec:6.2f}, time: {iter_time:8.6f}, total_time: {total_time:8.1f}, mlm_acc: {mlm_acc:8.5f}, nsp_acc: {nsp_acc:8.5f}")

        start_all = time.time()

        train_saver = train.saver["train_saver"]
        best_saver = train.saver["best_saver"]
        # We initialize the best loss to a super large value
        best_total_loss = 1e10
        best_step = 0

        while step < steps:
            # Run Training
            learning_rate = LR.feed_dict_lr(step)
            try:
                batch_time, mlm_loss, nsp_loss, mlm_acc, nsp_acc = training_step(train, learning_rate)
            except tf.errors.OpError as e:
                raise tf.errors.ResourceExhaustedError(e.node_def, e.op, e.message)

            epoch = float(opts["total_batch_size"] * i) / total_independent_samples

            batch_time /= iterations_per_step

            if step != 0:
                batch_times.append([batch_time])

            if step == 1:
                poplar_compile_time = time.time() - start_all
                poplar_summary = tf.Summary()
                poplar_summary.value.add(tag='poplar/compile_time', simple_value=poplar_compile_time)
                summary_writer.add_summary(poplar_summary)

            # Print loss
            if step % opts['steps_per_logs'] == 0:
                if len(batch_times) != 0:
                    avg_batch_time = np.mean(batch_times)
                else:
                    avg_batch_time = batch_time

                samples_per_sec = opts['total_batch_size'] / avg_batch_time

                # flush times every time it is reported
                batch_times.clear()

                total_time = time.time() - start_all

                stats = OrderedDict([
                    ('step', step),
                    ('iteration', i),
                    ('epoch', epoch),
                    ('lr', learning_rate),
                    ('mlm_loss', mlm_loss),
                    ('nsp_loss', nsp_loss),
                    ('mlm_acc', mlm_acc),
                    ('nsp_acc', nsp_acc),
                    ('iter_time', avg_batch_time),
                    ('samples_per_sec', samples_per_sec),
                    ('total_time', total_time),
                ])

                logger.info(print_format.format(**stats))
                bert_logging.write_to_csv(
                    stats, i == 0, True, opts['logs_path'])

                sys_summary = tf.Summary()
                sys_summary.value.add(tag='perf/throughput_samples_per_second', simple_value=samples_per_sec)
                sys_summary.value.add(tag='perf/average_batch_time', simple_value=avg_batch_time)
                summary_writer.add_summary(sys_summary, step)

            # Log training statistics
            train_summary = tf.Summary()
            train_summary.value.add(tag='epoch', simple_value=epoch)
            train_summary.value.add(tag='loss/MLM', simple_value=mlm_loss)
            train_summary.value.add(tag='loss/NSP', simple_value=nsp_loss)
            train_summary.value.add(tag='accuracy/MLM', simple_value=mlm_acc)
            train_summary.value.add(tag='accuracy/NSP', simple_value=nsp_acc)
            train_summary.value.add(tag='defaultLearningRate', simple_value=learning_rate)
            train_summary.value.add(tag='samples', simple_value=step*opts['batches_per_step']*opts['total_batch_size'])
            summary_writer.add_summary(train_summary, step)
            summary_writer.flush()

            if step % ckpt_per_step == 0 and step:
                filepath = train_saver.save(train.session, save_path=opts["checkpoint_path"], global_step=step)
                logger.info("Saved checkpoint to {}".format(filepath))

                if not opts['wandb']:
                    bert_logging.save_model_statistics(filepath, summary_writer, step)

            # Mechanism to checkpoint the best model.
            # set opts["best_ckpt_min_steps"] to 0 to disable
            if best_total_loss > mlm_loss + nsp_loss and step - best_step > opts["best_ckpt_min_steps"] and opts["best_ckpt_min_steps"]:
                best_total_loss = mlm_loss + nsp_loss
                best_step = step
                filepath = best_saver.save(train.session, save_path=opts["checkpoint_path"]+'_best', global_step=step)
                logger.info("Saved Best checkpoint to {}".format(filepath))

            i += iterations_per_step
            step += 1

        # --------------- LAST CHECKPOINT ----------------
        filepath = train_saver.save(train.session, save_path=opts["checkpoint_path"]+'_last', global_step=step)
        logger.info("Final model saved to to {}".format(filepath))

        # --------------- CLEANUP ----------------
        train.session.close()


def add_main_arguments(parser):
    group = parser.add_argument_group('Main')
    group.add_argument('--help', action='store_true', default=False,
                       help="Display help.")
    group.add_argument('--task', type=str, choices=['pretraining'],
                       help="Type of NLP task.")
    group.add_argument('--config', type=str,
                       help='BERT configuration file in JSON format.')
    group.add_argument('--restore_path', type=str, default=None,
                       help='Path to directory containing the checkpoint to restore.')
    group.add_argument('--start-from-ckpt', type=str, default=None,
                       help='Initial checkpoint from where we want to initialise the training.')
    return parser


def add_training_arguments(parser):

    pipeline_schedules_available = [
        p.name for p in ipu.ops.pipelining_ops.PipelineSchedule]
    recomputation_mode_available = [
        p.name for p in ipu.ops.pipelining_ops.RecomputationMode
    ]

    tr_group = parser.add_argument_group('Training')
    tr_group.add_argument('--batch-size', type=int,
                          help="Set batch-size for training graph")
    tr_group.add_argument('--base-learning-rate', type=float, default=2e-5,
                          help="Base learning rate exponent (2**N). blr = lr /  bs")
    tr_group.add_argument('--epochs', type=float, default=300,
                          help="Number of training epochs")
    tr_group.add_argument('--steps', type=int, default=0,
                          help="Number of steps after which we stop training.")
    tr_group.add_argument('--loss-scaling', type=float, default=1,
                          help="Loss scaling factor")
    tr_group.add_argument('--steps-per-ckpts', type=int, default=10,
                          help="Steps per checkpoints")
    tr_group.add_argument('--replicas', type=int, default=1,
                          help="Replicate graph over N workers to increase batch to batch-size*N")
    tr_group.add_argument('--pipeline-depth', type=int, default=1,
                          help="Depth of pipeline to use.")
    tr_group.add_argument('--pipeline-schedule', type=str, default='Grouped',
                          choices=pipeline_schedules_available, help="Pipelining scheduler.")
    tr_group.add_argument('--recomputation-mode', type=str, default="RecomputeAndBackpropagateInterleaved",
                          choices=recomputation_mode_available)
    tr_group.add_argument('--optimiser', type=str, default="SGD", choices=['SGD', 'momentum', 'lamb', 'adamw'],
                          help="Which optimiser to use for the optimisation of the neural network. Choices are: SGD, momentum, lamb, adamw. Default: SGD.")
    tr_group.add_argument('--momentum', type=float, default=0.984375,
                          help="Momentum coefficient.")
    tr_group.add_argument('--lr-schedule', default='natural_exponential',
                          choices=["custom", "polynomial_decay", "natural_exponential"],
                          help="Learning rate schedule function. Default: exponential")
    tr_group.add_argument('--warmup', type = float, default=0,
                          help="The Warmup period we want to use, default set to 0.")
    tr_group.add_argument('--offload-weight-update-variables', type=bool, default=True,
                          help="Offload variables used only during weight update to remote memory.")
    tr_group.add_argument('--partials-type', type=str, choices=['float', 'half'], default='half',
                          help="Storage type for the intermediate (partials) results of convolution and matmul operations")
    tr_group.add_argument('--optimiser-epsilon', type=float, default=1e-4,
                          help='Numerical Value to be used for numerical stability in the optimiser.')
    tr_group.add_argument('--increase-optimiser-precision', action='store_true',
                          help='In the LAMB optimiser, it performs more operations in fp32. This operation increase precision in the weight update but consumes more memory and reduce the Tput.')
    tr_group.add_argument('--optimiser_beta1', type=float, default=0.9,
                          help="Adam and LAMB beta_1 coefficient")
    tr_group.add_argument('--optimiser_beta2', type=float, default=0.999,
                          help="Adam and LAMB beta_2 coefficient")
    tr_group.add_argument('--use-nvlamb', action='store_true',
                          help="Flag to use the global normalisation for the gradients.")
    tr_group.add_argument('--use-debiasing', action='store_true',
                          help="Flag to use the de biasing for the momenta of LAMB")
    tr_group.add_argument('--wandb', action='store_true',
                          help="Enable logging and experiment tracking with Weights & Biases.")
    tr_group.add_argument('--seed', default=None,
                          help="Seed of the pseudo-random number generator used during traiing.")
    tr_group.add_argument('--do_train', default=True,
                          help="Configure the session for training.")
    tr_group.add_argument('--duplication_factor', default=5, help='The amount of duplication factor inside the dataset.')
    tr_group.add_argument('--ckpt-model-only', action='store_true',
                          help='If this flag is passed, the saver is going to checkpoint just the model and not the optimiser states.')
    tr_group.add_argument('--restore-optimiser-from-ckpt', action='store_true',
                          help='If this flag is passed the optimiser is going to be restored from the previous run, if the checkpoint has these value saved.')
    return parser


def set_training_defaults(opts):
    opts['name'] = 'BERT_' + opts['task']
    opts['total_batch_size'] = opts['batch_size'] * opts['pipeline_depth']*opts['replicas']
    logger.info(f"Total batch size: {opts['total_batch_size']}")


def add_ipu_arguments(parser):
    schedulers_available = ['Clustering', 'PostOrder', 'LookAhead', 'ShortestPath']

    group = parser.add_argument_group('IPU')
    group.add_argument('--precision', type=int, default=16, choices=[16, 32],
                       help="Precision of Ops(weights/activations/gradients).")
    group.add_argument('--batches_per_step', type=int, default=1,
                       help="Maximum number of batches to perform on the device before returning to the host.")
    group.add_argument('--available_memory_proportion', default=0.23, nargs='+',
                       help="Proportion of memory which is available for convolutions. Use a value of less than 0.6 "
                            "to reduce memory usage.")
    group.add_argument('--disable_graph_outlining', default=False, action="store_true",
                       help="Disable TensorFlow outlining optimisations.")
    group.add_argument("--embeddings_placement", type=str, default='two_ipus', choices=['two_ipus', 'same_ipu', 'same_as_hidden_layers'],
                       help="Placement of the embeddings layers in a pipelined model.\n"
                            "- 'two_ipus': Word embeddings on IPU 0, positional embeddings on IPU 1\n"
                            "- 'same_ipu: Word and positional embeddings in the same pipeline stage\n"
                            "- 'same_as_hidden_layers': Word and positional embeddings and hidden layers in the same pipeline stage")
    group.add_argument("--outline_hidden_layers", default=True, action="store_true",
                       help="Outline the hidden layers by using the @ipu.function decorator.")
    group.add_argument("--stochastic_rounding", default=True, action="store_true",
                       help="Use stochastic rounding.")
    group.add_argument("--fp_exceptions", default=False, action="store_true",
                       help="If true floating point invalid operations will cause an exception.")
    group.add_argument("--xla_recompute", default=True, action="store_true",
                       help="Recompute activations during backward pass")
    group.add_argument('--max-cross-replica-sum-buffer-size', type=int, default=10*1024*1024,
                       help="""The maximum number of bytes that can be waiting before a cross replica sum op is scheduled. [Default=10*1024*1024]""")
    group.add_argument('--scheduler', type=str, default='', choices=schedulers_available,
                       help="""Forces the compiler to use a specific scheduler when ordering the instructions.""")
    group.add_argument('--compile-only', action="store_true", default=False,
                       help="Configure Poplar to only compile the graph. This will not acquire any IPUs and thus facilitate profiling without using hardware resources.")
    group.add_argument('--logits_matmul_serialization_factor', type=int, default=4,
                       help="Number of smaller matrix multiplications the logits calculation is split into.")
    group.add_argument('--reduction-type', type=str, choices=['sum', 'mean'], default='sum',
                       help='The reduction type applied to the pipeline, the choice is between summation and mean.')
    group.add_argument('--weight-norm-clip', type=float, default=0.,
                       help='The value from which we want to clip the w_norm value, value of 0 is no weight clipping.')
    group.add_argument('--fix_synth_seed', default=False, action="store_true",
                       help='Set the seed for synthetic data generation, used in testing for consistency.')
    group.add_argument('--best-ckpt-min-steps', type=int, default=0,
                       help='Which is the minimal distance between two best checkpoints.')
    return parser


def add_bert_arguments(parser):
    group = parser.add_argument_group('BERT')
    group.add_argument('--vocab_size', type=int, default=31528)
    group.add_argument('--hidden_size', type=int, default=768)
    group.add_argument('--num_hidden_layers', type=int, default=12)
    group.add_argument('--num_attention_heads', type=int, default=12)
    group.add_argument('--hidden_act', type=str, default='gelu')
    group.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    group.add_argument('--attention_probs_dropout_prob', type=float, default=0.1)
    group.add_argument('--max_position_embeddings', type=int, default=512)
    group.add_argument('--type_vocab_size', type=int, default=16)
    group.add_argument('--initializer_range', type=float, default=0.02)
    group.add_argument('--hidden_layers_per_stage', default=1)
    group.add_argument('--max_predictions_per_seq', type=int, default=20)
    group.add_argument('--use_attention_projection_bias', type=bool, default=False,
                       help="Use biases in the projection dense layer.")
    group.add_argument('--use_cls_layer', type=bool, default=False,
                       help="Insert a dense layer before the output MLM logits.")
    group.add_argument('--use_qkv_bias', type=bool, default=False,
                       help="Use biases in the QKV dense layer.")
    return parser


def set_ipu_defaults(opts):
    poplar_version = os.popen('popc --version').read()
    logger.info(f"Poplar version: {poplar_version}")
    logger.info(f"Running on host: {gethostname()}")
    logger.info(f"Current date/time: {str(datetime.datetime.now())}")
    commit_hash = bert_logging.get_git_revision()
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


def create_command_line_parser():
    parser = argparse.ArgumentParser(
        description='BERT  Pretraining in TensorFlow',
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_main_arguments(parser)

    return parser


def create_all_options_parser():
    parser = argparse.ArgumentParser(
        description='BERT  Pretraining in TensorFlow',
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_main_arguments(parser)
    parser = dataset.add_arguments(parser)
    parser = add_training_arguments(parser)
    parser = add_ipu_arguments(parser)
    parser = bert_logging.add_arguments(parser)
    parser = add_bert_arguments(parser)
    return parser


def set_defaults(opts):
    dataset.set_defaults(opts)
    set_training_defaults(opts)
    set_ipu_defaults(opts)
    bert_logging.set_defaults(opts)


def get_bert_config_from_options(opts):
    return bert_ipu.BertConfig(
        opts['vocab_size'],
        opts['hidden_size'],
        opts['num_hidden_layers'],
        opts['num_attention_heads'],
        opts['hidden_act'],
        opts['hidden_dropout_prob'],
        opts['attention_probs_dropout_prob'],
        opts['max_position_embeddings'],
        opts['type_vocab_size'],
        opts['initializer_range'],
        opts['hidden_layers_per_stage'],
        opts['max_predictions_per_seq'],
        opts['use_attention_projection_bias'],
        opts['use_cls_layer'],
        opts['use_qkv_bias'],
        opts['outline_hidden_layers'],
        opts['logits_matmul_serialization_factor'],
        tf.float32 if opts["precision"] == 32 else tf.float16
    )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Parse command-line arguments
    command_line_parser = create_command_line_parser()
    all_options_parser = create_all_options_parser()

    known_command_line_args, unknown_command_line_args = command_line_parser.parse_known_args()

    if known_command_line_args.help:
        all_options_parser.print_help()
        sys.exit(0)

    # Parse options specified in the configuration file into
    config_file_path = known_command_line_args.config
    opts_from_config_file = bert_ipu.BertConfig.from_json_file(config_file_path)

    # Build the global options structure from the default options
    current_options = vars(all_options_parser.parse_args())

    # Overwrite global options by those specified in the config file.
    current_options.update(opts_from_config_file)
    options_namespace = argparse.Namespace(**current_options)
    # Overwrite with command-line arguments
    all_options_namespace = all_options_parser.parse_args(unknown_command_line_args, options_namespace)
    logger.info(f"Overwrite configuration parameters: {', '.join(unknown_command_line_args)}")

    # argparse.Namespace -> dict()
    opts = vars(all_options_namespace)

    # Group the options
    bert_config = get_bert_config_from_options(opts)

    set_defaults(opts)
    logger.info("Command line: " + ' '.join(sys.argv))
    logger.info("Option flags:\n" + json.dumps(opts, indent=1))
    train(bert_config, opts)
