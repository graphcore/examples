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

import argparse
import datetime
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import Counter, OrderedDict, namedtuple
from contextlib import ExitStack
from functools import partial
from shutil import copytree
from socket import gethostname

import numpy as np
import popdist
import popdist.tensorflow
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu import pipelining_ops
from tensorflow.python.ipu.config import DeviceConnectionType

import ipu_utils
import log
import modeling as bert_ipu
from bert_data import data_loader
from ipu_optimizer import get_optimizer
from log import logger
from loss_scaling_schedule import LossScalingScheduler
from lr_schedules import make_lr_schedule
from multi_stage_wrapper import (
    MultiStageEmbedding,
    get_split_embedding_stages,
    get_split_matmul_stages,
)
from options import make_global_options
from poplar_options import set_poplar_engine_options

import popdist
import popdist.tensorflow

GraphOps = namedtuple(
    "graphOps",
    [
        "graph",
        "session",
        "init",
        "ops",
        "placeholders",
        "iterator",
        "outfeed",
        "saver",
        "restore",
        "tvars",
    ],
)


def create_popdist_strategy():
    """
    Creates a distribution strategy for use with popdist. We use the
    Horovod-based PopDistStrategy. Horovod is used for the initial
    broadcast of the weights and when reductions are requested on the host.
    Imports are placed here so they are only done when required, as Horovod
    might not always be available.
    """

    from tensorflow.python.ipu.horovod import popdist_strategy

    hvd.init()

    # We add the IPU cross replica reductions explicitly in the IPUOptimizer,
    # so disable them in the PopDistStrategy.
    return popdist_strategy.PopDistStrategy(add_ipu_cross_replica_reductions=False)


def build_pretrain_pipeline_stages(model, bert_config, opts):
    """
    build pipeline stages according to "pipeline_stages" in config file
    """

    # flatten stages config into list of layers
    flattened_layers = []
    for stage in opts["pipeline_stages"]:
        flattened_layers.extend(stage)
    layer_counter = Counter(flattened_layers)
    assert layer_counter["hid"] == opts["num_hidden_layers"]
    assert layer_counter["emb"] == layer_counter["mlm"]
    # gradient_accumulation_count needs to be a multiple of stage_number*2
    # this is constrained by sdk
    assert opts["gradient_accumulation_count"] % (
        len(opts["pipeline_stages"]) * 2) == 0

    computational_stages = []
    if layer_counter["emb"] > 1:
        # support distribute embedding to multiple IPUs
        embedding = MultiStageEmbedding(
            embedding_size=bert_config.hidden_size,
            vocab_size=bert_config.vocab_size,
            initializer_range=bert_config.initializer_range,
            n_stages=layer_counter["emb"],
            matmul_serialize_factor=opts["matmul_serialize_factor"],
            dtype=bert_config.dtype,
        )
        embedding_stages = get_split_embedding_stages(
            embedding=embedding,
            split_count=layer_counter["emb"],
            bert_config=bert_config,
            micro_batch_size=opts["micro_batch_size"],
            seq_length=opts["seq_length"],
        )
        # masked lm better be on same ipu with embedding layer for saving
        # storage
        masked_lm_output_post_stages = get_split_matmul_stages(
            embedding=embedding,
            split_count=layer_counter["emb"],
            bert_config=bert_config,
        )
    else:
        embedding_stages = [model.embedding_lookup_layer]
        masked_lm_output_post_stages = [model.mlm_head]

    layers = {
        "emb": embedding_stages,
        "pos": model.embedding_postprocessor_layer,
        "hid": model.encoder,
        "mlm": masked_lm_output_post_stages,
        "nsp": model.get_next_sentence_output_layer,
    }
    stage_layer_list = []
    for stage in opts["pipeline_stages"]:
        func_list = []
        for layer in stage:
            # embedding layer and mlm layer can be splited to mutliple IPUs, so need to be dealt with separately
            if layer == "emb":
                func_list.append(embedding_stages[0])
                embedding_stages = embedding_stages[1:]
            elif layer == "mlm":
                func_list.append(masked_lm_output_post_stages[0])
                masked_lm_output_post_stages = masked_lm_output_post_stages[1:]
            else:
                func_list.append(layers[layer])
        stage_layer_list.append(func_list)
    computational_stages = ipu_utils.stages_constructor(
        stage_layer_list,
        ["learning_rate", "loss_scaling"],
        [
            "learning_rate",
            "loss_scaling",
            "mlm_loss",
            "nsp_loss",
            "mlm_acc",
            "nsp_acc",
        ],
    )

    return computational_stages


def build_network(
    infeed,
    outfeed,
    bert_config=None,
    opts=None,
    learning_rate=None,
    loss_scaling=None,
    is_training=True,
):

    # build model
    if opts["groupbert"]:
        logger.info(
            "************* Using GroupBERT model architecture *************")
        pipeline_model = bert_ipu.GroupBertModel(
            bert_config, is_training=is_training)
    else:
        pipeline_model = bert_ipu.BertModel(
            bert_config, is_training=is_training)

    # build stages & device mapping
    computational_stages = build_pretrain_pipeline_stages(
        pipeline_model,
        bert_config,
        opts,
    )
    device_mapping = opts["device_mapping"]

    logger.info(
        f"************* computational stages: *************\n{computational_stages}"
    )
    logger.info(
        f"************* device mapping: *************\n{device_mapping}")

    # define optimizer
    def optimizer_function(
        learning_rate, loss_scaling, mlm_loss, nsp_loss, mlm_acc, nsp_acc
    ):
        total_loss = mlm_loss + nsp_loss
        optimizer = get_optimizer(
            learning_rate, loss_scaling, opts["total_replicas"], opts
        )
        fp32_loss = tf.cast(total_loss, tf.float32) * loss_scaling
        return ipu.ops.pipelining_ops.OptimizerFunctionOutput(optimizer, fp32_loss)

    # Set IPU-specific available memory proportion
    if isinstance(opts["available_memory_proportion"], float):
        available_memory_proportion_list = [
            str(opts["available_memory_proportion"])
        ] * len(device_mapping)
    else:
        available_memory_proportion_list = [
            str(opts["available_memory_proportion"][device])
            for device in device_mapping
        ]

    if len(available_memory_proportion_list) != len(device_mapping):
        raise ValueError(
            "The available_memory_proportion list must be the same length as the number of stages in the pipeline."
        )

    options = [
        ipu.pipelining_ops.PipelineStageOptions(
            matmul_options={
                "availableMemoryProportion": amp,
                "partialsType": opts["partials_type"],
            }
        )
        for amp in available_memory_proportion_list
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
                                                       repeat_count=opts['device_iterations'],
                                                       inputs=[
                                                           learning_rate, loss_scaling],
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
                                                       accumulate_outfeed=True,
                                                       replicated_optimizer_state_sharding=opts[
                                                           'replicated_tensor_sharding'],
                                                       name="Pipeline")
    else:
        pipeline_ops = ipu.ops.pipelining_ops.pipeline(computational_stages=computational_stages,
                                                       gradient_accumulation_count=int(
                                                           opts['gradient_accumulation_count']),
                                                       repeat_count=opts['device_iterations'],
                                                       inputs=[
                                                           learning_rate, loss_scaling],
                                                       infeed_queue=infeed,
                                                       outfeed_queue=outfeed,
                                                       device_mapping=device_mapping,
                                                       forward_propagation_stages_poplar_options=options,
                                                       backward_propagation_stages_poplar_options=options,
                                                       offload_weight_update_variables=opts["variable_offloading"],
                                                       pipeline_schedule=pipeline_schedule,
                                                       recomputation_mode=ipu.ops.pipelining_ops.RecomputationMode[
                                                           opts['recomputation_mode']],
                                                       replicated_optimizer_state_sharding=opts[
                                                           'replicated_tensor_sharding'],
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
def training_step_with_infeeds_and_outfeeds(
    train_iterator,
    outfeed_queue,
    bert_config,
    opts,
    learning_rate,
    loss_scaling,
    is_training,
):
    """
    Training step that uses an infeed loop with outfeeds. This runs 'iterations_per_step' steps per session call. This leads to
    significant speed ups on IPU. Not compatible with running on CPU or GPU.
    """

    if opts["gradient_accumulation_count"] > 1:
        training_step = partial(
            build_network,
            infeed=train_iterator,
            outfeed=outfeed_queue,
            bert_config=bert_config,
            opts=opts,
            learning_rate=learning_rate,
            loss_scaling=loss_scaling,
            is_training=is_training,
        )

    return ipu.ipu_compiler.compile(training_step, [])


def build_graph(opts, is_training=True):
    train_graph = tf.Graph()
    strategy = None

    if opts["use_popdist"]:
        strategy = create_popdist_strategy()

    with train_graph.as_default(), ExitStack() as stack:
        if strategy:
            stack.enter_context(strategy.scope())

        if opts["groupbert"]:
            bert_config = bert_ipu.BertConfig.from_dict(
                opts, config=bert_ipu.GroupBertConfig(vocab_size=None)
            )
        else:
            bert_config = bert_ipu.BertConfig.from_dict(
                opts, config=bert_ipu.BertConfig(vocab_size=None)
            )

        bert_config.dtype = tf.float32 if opts["precision"] == "32" else tf.float16

        # define placeholders
        placeholders = {
            "learning_rate": tf.placeholder(tf.float32, shape=[]),
            "loss_scaling": tf.placeholder(tf.float32, shape=[]),
        }
        learning_rate = placeholders["learning_rate"]
        loss_scaling = placeholders["loss_scaling"]

        # define input, datasets must be defined outside the ipu device scope.
        train_iterator = ipu.ipu_infeed_queue.IPUInfeedQueue(
            data_loader.load(opts, is_training=is_training)
        )
        # define output
        outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

        # building networks with pipeline
        def bert_net():
            return build_network(
                train_iterator,
                outfeed_queue,
                bert_config,
                opts,
                learning_rate,
                loss_scaling,
                is_training,
            )

        with ipu.scopes.ipu_scope("/device:IPU:0"):
            train = training_step_with_infeeds_and_outfeeds(
                train_iterator,
                outfeed_queue,
                bert_config,
                opts,
                learning_rate,
                loss_scaling,
                is_training,
            )

        # get result from outfeed queue
        outfeed = outfeed_queue.dequeue()

        if strategy:
            # Take the mean of all the outputs across the distributed workers
            outfeed = [strategy.reduce(
                tf.distribute.ReduceOp.MEAN, v) for v in outfeed]

        if opts["distributed_worker_index"] == 0 or opts["log_all_workers"]:
            log.print_trainable_variables(opts)

        model_and_optimiser_variables = tf.global_variables()
        model_variables = tf.trainable_variables() + tf.get_collection(
            tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES
        )
        restore = tf.train.Saver(
            var_list=model_and_optimiser_variables
            if opts["restore_optimiser_from_checkpoint"]
            else model_variables
        )

        train_saver = tf.train.Saver(
            var_list=model_and_optimiser_variables
            if opts["save_optimiser_to_checkpoint"]
            else model_variables,
            max_to_keep=opts["max_to_keep"],
        )

        ipu.utils.move_variable_initialization_to_cpu()
        train_init = tf.global_variables_initializer()
        tvars = tf.trainable_variables()

    # calculate the number of required IPU
    num_ipus = (max(opts["device_mapping"]) + 1) * opts["replicas"]
    num_ipus = ipu_utils.next_power_of_two(num_ipus)

    ipu_config = ipu_utils.get_config(
        fp_exceptions=opts["fp_exceptions"],
        enable_recomputation=opts["enable_recomputation"],
        disable_graph_outlining=False,
        num_required_ipus=num_ipus,
        enable_stochastic_rounding=opts["stochastic_rounding"],
        minimum_remote_tensor_size=opts["min_remote_tensor_size"],
        max_cross_replica_sum_buffer_size=opts["max_cross_replica_sum_buffer_size"],
        max_reduce_scatter_buffer_size=opts["max_reduce_scatter_buffer_size"],
        scheduler_selection=opts["scheduler"],
        compile_only=opts["compile_only"],
        ipu_id=opts["select_ipu"],
    )

    if opts["use_popdist"]:
        ipu_config = popdist.tensorflow.set_ipu_config(
            ipu_config, opts["shards"], configure_device=False
        )

    # Do not acquire a device, compile only.
    if opts["compile_only"]:
        # Enforce using a exe cache dir, defaulting if not given
        if "TF_POPLAR_FLAGS" in os.environ:
            if "--executable_cache_path" not in os.environ["TF_POPLAR_FLAGS"]:
                print("Warning: --executable_cache_path in TF_POPLAR_FLAGS (for 'poprun --mpi_local_args') not set. Setting to default path: ./tmp/tf_cache/")
                os.environ["TF_POPLAR_FLAGS"] = "--executable_cache_path=/tmp/tf_cache"

        # Sometimes TF_POPLAR_FLAGS might not even exist
        else:
            print(
                "Warning: TF_POPLAR_FLAGS environment variable (for 'poprun --mpi_local_args') not set. --executable_cache_path must be defined when using --compile-only. Setting to default path: ./tmp/tf_cache/"
            )
            os.environ["TF_POPLAR_FLAGS"] = "--executable_cache_path=/tmp/tf_cache"

    ipu_config.configure_ipu_system()

    train_sess = tf.Session(graph=train_graph)

    return GraphOps(
        train_graph,
        train_sess,
        train_init,
        [train],
        placeholders,
        train_iterator,
        outfeed,
        train_saver,
        restore,
        tvars,
    )


def training_step(train, learning_rate, loss_scaling):
    start = time.time()
    _ = train.session.run(
        train.ops,
        feed_dict={
            train.placeholders["learning_rate"]: learning_rate,
            train.placeholders["loss_scaling"]: loss_scaling,
        },
    )
    batch_time = time.time() - start
    if not os.environ.get(
        "TF_POPLAR_FLAGS"
    ) or "--use_synthetic_data" not in os.environ.get("TF_POPLAR_FLAGS"):
        (
            _learning_rate,
            _loss_scaling_,
            _mlm_loss,
            _nsp_loss,
            _mlm_acc,
            _nsp_acc,
        ) = train.session.run(train.outfeed)
        # We need to divide explicitly by the accumulated gradient since it gets accumulated implicitly inside the pipeline.
        mlm_loss = np.mean(_mlm_loss) / opts["gradient_accumulation_count"]
        nsp_loss = np.mean(_nsp_loss) / opts["gradient_accumulation_count"]
        mlm_acc = np.mean(_mlm_acc) / opts["gradient_accumulation_count"]
        nsp_acc = np.mean(_nsp_acc) / opts["gradient_accumulation_count"]
        if mlm_acc == -1 and nsp_acc == -1:
            # If they are both disabled then it is worth to put Nan instead
            mlm_acc = np.nan
            nsp_acc = np.nan
    else:
        mlm_loss, nsp_loss = 0, 0
        mlm_acc, nsp_acc = 0, 0
    return batch_time, mlm_loss, nsp_loss, mlm_acc, nsp_acc


def train(opts):
    # --------------- OPTIONS ---------------------
    total_samples = data_loader.get_dataset_files_count(opts, is_training=True)
    opts["dataset_repeat"] = math.ceil(
        (opts["num_train_steps"] * opts["global_batch_size"]) / total_samples
    )

    total_samples_per_epoch = total_samples / opts["duplicate_factor"]
    logger.info(f"Total samples for each epoch {total_samples_per_epoch}")
    logger.info(f"Global batch size {opts['global_batch_size']}")
    steps_per_epoch = total_samples_per_epoch // opts["global_batch_size"]
    logger.info(f"Total steps for each epoch {steps_per_epoch}")

    steps_per_logs = math.ceil(
        opts["steps_per_logs"] / opts['device_iterations']) * opts['device_iterations']
    steps_per_tensorboard = math.ceil(
        opts["steps_per_tensorboard"] / opts['device_iterations']) * opts['device_iterations']
    steps_per_ckpts = math.ceil(
        opts["steps_per_ckpts"] / opts['device_iterations']) * opts['device_iterations']
    logger.info(f"Checkpoint will be saved every {steps_per_ckpts} steps.")

    total_steps = (opts["num_train_steps"] //
                   opts['device_iterations']) * opts['device_iterations']
    logger.info(f"{opts['device_iterations']} steps will be run for ipu to host synchronization once, it should be divided by num_train_steps, so num_train_steps will limit to {total_steps}.", opts)

    # learning rate strategy
    lr_schedule_name = opts["lr_schedule"]
    logger.info(f"Using learning rate schedule {lr_schedule_name}")
    learning_rate_schedule = make_lr_schedule(
        lr_schedule_name, opts, total_steps)

    # variable loss scaling
    loss_scaling_schedule = LossScalingScheduler(
        opts["loss_scaling"], opts["loss_scaling_by_step"]
    )

    # -------------- BUILD TRAINING GRAPH ----------------
    train = build_graph(opts, is_training=True)
    train.session.run(train.init)
    train.session.run(train.iterator.initializer)

    is_main_worker = opts["distributed_worker_index"] == 0

    step = 0
    # -------------- SAVE AND RESTORE --------------
    if opts["restore_dir"]:
        restore_path = opts["restore_dir"]
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
        source_path = os.path.join(opts["restore_dir"], "/event")
        target_path = os.path.join(opts["save_path"], "/event")
        if os.path.isdir(source_path):
            copytree(source_path, target_path)
    else:
        if opts["init_checkpoint"]:
            train.saver.restore(train.session, opts["init_checkpoint"])
            logger.info(
                f'Init Model from checkpoint {opts["init_checkpoint"]}')

    if opts["save_path"]:
        file_path = train.saver.save(
            train.session, opts["checkpoint_path"], global_step=0
        )
        logger.info(f"Saved checkpoint to {file_path}")

    # Initialise Weights & Biases if available
    if opts["wandb"] and is_main_worker:
        import wandb

        wandb.init(project="tf-bert", sync_tensorboard=True,
                   name=opts["wandb_name"])
        wandb.config.update(opts)

    # Tensorboard logs path
    log_path = os.path.join(opts["logs_path"], "event")
    logger.info("Tensorboard event file path {}".format(log_path))
    summary_writer = tf.summary.FileWriter(
        log_path, train.graph, session=train.session)

    # End to avoid any training if compile only mode
    if opts["compile_only"]:

        # single warm up step without weight update or training
        # Graph gets compiled in here
        compilation_time, _, _, _, _ = training_step(train, 0, 0)

        print(
            "Training graph successfully compiled. Exiting as --compile-only was passed."
        )

        # Copying these from below, adding compile time to summary
        poplar_summary = tf.Summary()
        poplar_summary.value.add(
            tag="poplar/compile_time", simple_value=compilation_time
        )
        summary_writer.add_summary(poplar_summary)
        summary_writer.flush()

        logger.info("Compile time: {}".format(compilation_time))

        sys.exit(0)

    # ------------- TRAINING LOOP ----------------
    print_format = "step: {step:6d}, epoch: {epoch:6.2f}, lr: {lr:6.7f}, mlm_loss: {mlm_loss:6.3f}, nsp_loss: {nsp_loss:6.3f},\
        mlm_acc: {mlm_acc:6.5f}, nsp_acc: {nsp_acc:6.5f}, samples/sec: {samples_per_sec:6.2f}, time: {iter_time:8.6f}, total_time: {total_time:8.1f}"
    learning_rate = mlm_loss = nsp_loss = 0
    start_all = time.time()

    try:
        while step < total_steps:
            learning_rate = learning_rate_schedule.get_at_step(step)
            loss_scaling = loss_scaling_schedule.get_at_step(step)
            try:
                (
                    batch_time,
                    mlm_loss,
                    nsp_loss,
                    mlm_acc,
                    nsp_acc,
                ) = training_step(train, learning_rate, loss_scaling)
            except tf.errors.OpError as e:
                raise tf.errors.ResourceExhaustedError(
                    e.node_def, e.op, e.message)

            batch_time /= opts['device_iterations']

            is_log_step = (step % steps_per_logs == 0)
            is_save_tensorboard_step = (steps_per_tensorboard > 0 and (
                step % steps_per_tensorboard == 0))
            is_save_ckpt_step = (step and (
                step % steps_per_ckpts == 0 or step == total_steps - opts['device_iterations']))

            if step == 1 and (is_main_worker or opts["log_all_workers"]):
                poplar_compile_time = time.time() - start_all
                logger.info(f"Poplar compile time: {poplar_compile_time:.2f}s")
                poplar_summary = tf.Summary()
                poplar_summary.value.add(
                    tag="poplar/compile_time", simple_value=poplar_compile_time
                )
                summary_writer.add_summary(poplar_summary)

            if is_log_step:
                total_time = time.time() - start_all
                epoch = step / steps_per_epoch
                stats = OrderedDict(
                    [
                        ("step", step),
                        ("epoch", epoch),
                        ("lr", learning_rate),
                        ("loss_scaling", loss_scaling),
                        ("mlm_loss", mlm_loss),
                        ("nsp_loss", nsp_loss),
                        ("mlm_acc", mlm_acc),
                        ("nsp_acc", nsp_acc),
                        ("iter_time", batch_time),
                        (
                            "samples_per_sec",
                            opts["global_batch_size"] / batch_time,
                        ),
                        ("total_time", total_time),
                    ]
                )

                logger.info(print_format.format(**stats))

            # Log training statistics
            train_summary = tf.Summary()
            train_summary.value.add(tag="epoch", simple_value=epoch)
            train_summary.value.add(tag="loss/MLM", simple_value=mlm_loss)
            train_summary.value.add(tag="loss/NSP", simple_value=nsp_loss)
            train_summary.value.add(tag="accuracy/MLM", simple_value=mlm_acc)
            train_summary.value.add(tag="accuracy/NSP", simple_value=nsp_acc)
            train_summary.value.add(
                tag="learning_rate", simple_value=learning_rate)
            train_summary.value.add(
                tag="loss_scaling", simple_value=loss_scaling)
            train_summary.value.add(
                tag="samples_per_sec",
                simple_value=opts["global_batch_size"] / batch_time,
            )
            train_summary.value.add(
                tag='samples', simple_value=step*opts['device_iterations'] * opts['global_batch_size'])
            summary_writer.add_summary(train_summary, step)
            summary_writer.flush()

            if is_save_ckpt_step or is_save_tensorboard_step:
                if is_main_worker:
                    file_path = train.saver.save(
                        train.session,
                        opts["checkpoint_path"],
                        global_step=step,
                    )
                    logger.info(f"Saved checkpoint to {file_path}")

                    if is_save_tensorboard_step:
                        log.save_model_statistics(
                            file_path, summary_writer, step)

                if opts["use_popdist"]:
                    ipu_utils.barrier()

            step += opts['device_iterations']
    finally:
        train.session.close()


def set_distribution_defaults(opts):
    if opts["use_popdist"]:
        opts["distributed_worker_count"] = popdist.getNumInstances()
        opts["distributed_worker_index"] = popdist.getInstanceIndex()
    else:
        opts["distributed_worker_count"] = 1
        opts["distributed_worker_index"] = 0

    if opts["distributed_worker_index"] != 0 and not opts["log_all_workers"]:
        logger.setLevel(logging.ERROR)


def set_training_defaults(opts):
    # Automatic pipeline depth counter
    if opts["global_batch_size"]:
        gradients_to_accumulate = opts["global_batch_size"] // (
            opts["total_replicas"] * opts["micro_batch_size"]
        )
        divisor = len(opts["pipeline_stages"]) * 2
        # We need then to fix the gradient_to_accumulate according to the pipeline
        gradients_to_accumulate = divisor * \
            (1 + gradients_to_accumulate // divisor)
        if (
            opts["gradient_accumulation_count"] and opts["gradient_accumulation_count"] != gradients_to_accumulate
        ):
            logger.error(
                "Passed a gradient to accumulate and a global batch size. Disable one of them to run."
            )
            sys.exit(os.EX_OK)
        opts["gradient_accumulation_count"] = gradients_to_accumulate
        # We update the global_batch_size
        proposed_global_batch_size = (
            opts["gradient_accumulation_count"] *
            opts["total_replicas"] * opts["micro_batch_size"]
        )
        if proposed_global_batch_size != opts["global_batch_size"]:
            logger.info(
                "Changing the global batch size to match the pipeline requirements."
            )
            opts["global_batch_size"] = proposed_global_batch_size
    else:
        opts["global_batch_size"] = (
            opts["micro_batch_size"] *
            opts["gradient_accumulation_count"] * opts["total_replicas"]
        )

    opts["compute_acc"] = not opts["disable_acc"]
    if opts["disable_acc"]:
        logger.info(
            "Disabling computation of the accuracies. Just the losses will be reported."
        )


def set_ipu_defaults(opts):
    poplar_version = os.popen("popc --version").read()
    opts["poplar_version"] = poplar_version
    logger.info(f"Running on host: {gethostname()}")
    logger.info(f"Current date/time: {str(datetime.datetime.now())}")
    commit_hash = log.get_git_revision()
    logger.info(f"Code revision: {commit_hash}")
    seed = opts["seed"]
    logger.info(f"Pseudo-random number generator seed specified: f{seed}")
    random.seed(seed)
    # Set other seeds to different values for extra safety
    tf.set_random_seed(random.randint(0, 2**32 - 1))
    np.random.seed(random.randint(0, 2**32 - 1))
    ipu.utils.reset_ipu_seed(
        random.randint(-(2**16), 2**16 - 1),
        experimental_identical_replicas=opts["ipu_replica_identical_seed"],
    )


def set_defaults(opts):
    data_loader.set_defaults(opts)
    set_distribution_defaults(opts)
    set_training_defaults(opts)
    set_ipu_defaults(opts)
    log.set_defaults(opts)


def add_pretraining_options(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Pretraining options")
    # Add pretraining-specific command line options here.
    return parser


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)

    opts = make_global_options([add_pretraining_options])

    opts["shards"] = ipu_utils.next_power_of_two(
        max(opts["device_mapping"]) + 1)

    if opts["seed"] is None:
        # Seed the various random sources
        opts["seed"] = random.randint(0, 2**32 - 1)
        logger.info(f"Using random number generated seed: f{opts['seed']}")

    if popdist.isPopdistEnvSet():
        opts["use_popdist"] = True
        opts["replicas"] = popdist.getNumLocalReplicas()
        opts["total_replicas"] = popdist.getNumTotalReplicas()
        if opts["compile_only"]:
            opts["select_ipu"] = None
        else:
            opts["select_ipu"] = popdist.getDeviceId()
    else:
        opts["use_popdist"] = False
        opts["total_replicas"] = opts["replicas"]
        opts["select_ipu"] = None

    set_defaults(opts)

    set_poplar_engine_options(
        execution_profile=opts["execution_profile"],
        memory_profile=opts["memory_profile"],
        profile_dir=str(opts["profile_dir"]),
        sync_replicas_independently=opts["replicas"] > 1 and opts["sync_replicas_independently"],
        synthetic_data=opts["synthetic_data"],
        tensorflow_progress_bar=opts["progress_bar"],
        ipu_replica_identical_seed=opts["ipu_replica_identical_seed"],
    )

    poplar_options = os.getenv("POPLAR_ENGINE_OPTIONS", "unset")
    logger.info(f"Poplar options: {poplar_options}")
    logger.info("Command line: " + " ".join(sys.argv))
    if opts["use_popdist"] and opts["log_all_workers"]:
        option_string = f"Option flags for worker {opts['distributed_worker_index']}:\n"
    else:
        option_string = f"Option flags:\n"
    logger.info(option_string +
                json.dumps(OrderedDict(sorted(opts.items())), indent=1))

    # Start training
    train(opts)
