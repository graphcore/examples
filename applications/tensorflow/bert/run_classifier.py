#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
#
# Copyright 2018 The Google AI Language Team Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import chain
from typing import List
from options import make_global_options
from lr_schedules import make_lr_schedule
from log import logger
from ipu_utils import get_config, stages_constructor
from ipu_optimizer import get_optimizer
from bert_data import squad_results, tokenization
from bert_data import glue as glue_data
from bert_data import squad as squad_data
from bert_data import data_loader
import modeling as bert_ipu
import log
from tensorflow.contrib.data import map_and_batch
from tensorflow.python.ipu.utils import reset_ipu_seed
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ipu import ipu_infeed_queue, ipu_outfeed_queue
from tensorflow.python import ipu
import tensorflow.compat.v1 as tf
import numpy as np
from socket import gethostname
from collections import OrderedDict, deque, namedtuple
import time
import sys
import subprocess
import re
import random
import math
import json
import datetime
import argparse


import collections
import csv
import os
import modeling
# import optimization
import bert_data.tokenization as tokenization
import tensorflow as tf
import scipy.stats as sci

# Graph data structure
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


def build_squad_pipeline_stages(model, bert_config, opts, is_training):
    """
    build pipeline stages according to "pipeline_stages" in config file
    """

    # flatten stages config into list of layers
    flattened_layers = []
    for stage in opts['pipeline_stages']:
        flattened_layers.extend(stage)
    layer_counter = collections.Counter(flattened_layers)
    assert layer_counter['hid'] == opts['num_hidden_layers']
    assert layer_counter['emb'] == 1

    # gradient_accumulation_count need to be a multiple of stage_number*2
    # this is constrained by sdk
    assert opts['gradient_accumulation_count'] % (
        len(opts['pipeline_stages'])*2) == 0
    layers = {
        'emb': model.embedding_lookup_layer,
        'pos': model.embedding_postprocessor_layer,
        'hid': model.encoder,
        'glu': model.get_glue_output_layer,
        'glu_reg': model.get_glue_regression_layer
    }

    stage_layer_list = []
    for stage in opts['pipeline_stages']:
        func_list = []
        for layer in stage:
            func_list.append(layers[layer])
        stage_layer_list.append(func_list)
    if is_training:
        if opts['task_type'] == 'regression':
            computational_stages = stages_constructor(
                stage_layer_list, ['learning_rate', 'label_ids'], ['learning_rate', 'total_loss', 'per_example_loss', 'logits', 'label_ids'])
        else:
            computational_stages = stages_constructor(
                stage_layer_list, ['learning_rate', 'label_ids'], ['learning_rate', 'total_loss', 'per_example_loss', 'logits', 'preds', 'acc'])
    else:
        if opts['task_type'] == 'regression':
            computational_stages = stages_constructor(
                stage_layer_list, [], ['learning_rate', 'total_loss', 'per_example_loss', 'logits', 'label_ids'])
        else:
            computational_stages = stages_constructor(
                stage_layer_list, [], ['learning_rate', 'total_loss', 'per_example_loss', 'logits', 'preds', 'acc'])

    return computational_stages


def build_network(infeed,
                  outfeed,
                  iterations_per_step=1,
                  bert_config=None,
                  opts=None,
                  learning_rate=None,
                  is_training=True):
    # build model
    if opts["groupbert"]:
        logger.info(f"************* Using GroupBERT model architecture *************")
        pipeline_model = bert_ipu.GroupBertModel(bert_config, is_training=is_training)
    else:
        pipeline_model = bert_ipu.BertModel(bert_config, is_training=is_training)

    # build stages & device mapping
    computational_stages = build_squad_pipeline_stages(
        pipeline_model, bert_config, opts, is_training)
    device_mapping = opts['device_mapping']
    logger.info(
        f"************* computational stages: *************\n{computational_stages}")
    logger.info(
        f"************* device mapping: *************\n{device_mapping}")

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
        matmul_options={"availableMemoryProportion": str(
            opts["available_memory_proportion"]), "partialsType": opts["partials_type"]},
        convolution_options={"partialsType": opts["partials_type"]})] * len(computational_stages)

    if is_training:
        # define optimizer
        def optimizer_function(learning_rate, total_loss, *args):
            optimizer = get_optimizer(
                learning_rate, opts['loss_scaling'], opts['replicas'], opts)
            if opts["replicas"] > 1:
                optimizer = ipu.optimizers.cross_replica_optimizer.CrossReplicaOptimizer(
                    optimizer)
            return pipelining_ops.OptimizerFunctionOutput(optimizer, total_loss * opts['loss_scaling'])

        return pipelining_ops.pipeline(computational_stages=computational_stages,
                                       gradient_accumulation_count=opts['gradient_accumulation_count'],
                                       repeat_count=iterations_per_step,
                                       inputs=[learning_rate],
                                       infeed_queue=infeed,
                                       outfeed_queue=outfeed,
                                       device_mapping=device_mapping,
                                       forward_propagation_stages_poplar_options=options,
                                       backward_propagation_stages_poplar_options=options,
                                       offload_weight_update_variables=opts['variable_offloading'],
                                       optimizer_function=optimizer_function,
                                       recomputation_mode=ipu.ops.pipelining_ops.RecomputationMode[
                                           opts['recomputation_mode']],
                                       name="Pipeline")
    else:
        return pipelining_ops.pipeline(computational_stages=computational_stages,
                                       gradient_accumulation_count=opts['gradient_accumulation_count'],
                                       repeat_count=iterations_per_step,
                                       inputs=[],
                                       infeed_queue=infeed,
                                       outfeed_queue=outfeed,
                                       device_mapping=device_mapping,
                                       forward_propagation_stages_poplar_options=options,
                                       backward_propagation_stages_poplar_options=options,
                                       offload_weight_update_variables=opts['variable_offloading'],
                                       name="Pipeline")


def build_graph(opts, iterations_per_step=1, is_training=True):

    train_graph = tf.Graph()
    with train_graph.as_default():
        if opts["groupbert"]:
            bert_config = bert_ipu.BertConfig.from_dict(
                opts, config=bert_ipu.GroupBertConfig(vocab_size=None))
        else:
            bert_config = bert_ipu.BertConfig.from_dict(
                opts, config=bert_ipu.BertConfig(vocab_size=None))
        bert_config.dtype = tf.float32 if opts["precision"] == '32' else tf.float16
        placeholders = dict()

        if is_training:
            placeholders['learning_rate'] = tf.placeholder(
                bert_config.dtype, shape=[])
            learning_rate = placeholders['learning_rate']
        else:
            learning_rate = None

        # Need to load the Glue File here
        label_list = opts["pass_in"][1]
        bert_config.num_lables = len(label_list)
        if opts['do_training'] and opts['current_mode'] == 'train':
            input_file = os.path.join(opts["output_dir"], f"train_{opts['task_type']}.tf_record")
        elif opts['do_eval'] and opts['current_mode'] == 'eval':
            input_file = os.path.join(opts["output_dir"], f"eval_{opts['task_type']}.tf_record")
        elif opts['do_predict'] and opts['current_mode'] == 'predict':
            input_file = os.path.join(opts["output_dir"], f"predict_{opts['task_type']}.tf_record")
        else:
            raise NotImplementedError()


        opts['input_file'] = input_file
        opts['drop_remainder'] = True

        train_iterator = ipu_infeed_queue.IPUInfeedQueue(data_loader.load(opts, is_training=is_training))
        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

        def bert_net():
            return build_network(train_iterator,
                                 outfeed_queue,
                                 iterations_per_step,
                                 bert_config,
                                 opts,
                                 learning_rate,
                                 is_training)

        with ipu_scope('/device:IPU:0'):
            train = ipu.ipu_compiler.compile(bert_net, [])

        outfeed = outfeed_queue.dequeue()

        log.print_trainable_variables(opts)

        restore = tf.train.Saver(var_list=tf.global_variables())
        train_saver = tf.train.Saver(max_to_keep=5)

        ipu.utils.move_variable_initialization_to_cpu()
        train_init = tf.global_variables_initializer()
        tvars = tf.trainable_variables()

    """calculate the number of required IPU"""
    num_ipus = (max(opts['device_mapping'])+1) * int(opts['replicas'])
    # The number of acquired IPUs must be the power of 2.
    if num_ipus & (num_ipus - 1) != 0:
        num_ipus = 2**int(math.ceil(math.log(num_ipus) / math.log(2)))
    ipu_config = get_config(fp_exceptions=opts["fp_exceptions"],
                            enable_recomputation=opts["enable_recomputation"],
                            disable_graph_outlining=False,
                            num_required_ipus=num_ipus,
                            enable_stochastic_rounding=opts['stochastic_rounding'],
                            max_cross_replica_sum_buffer_size=opts['max_cross_replica_sum_buffer_size'],
                            max_reduce_scatter_buffer_size=opts['max_reduce_scatter_buffer_size'],
                            scheduler_selection='CLUSTERING',
                            compile_only=False,
                            ipu_id=None,
                            available_memory_proportion=opts["available_memory_proportion"])

    ipu_config.configure_ipu_system()

    train_sess = tf.Session(graph=train_graph)

    return GraphOps(train_graph, train_sess, train_init, [train], placeholders, train_iterator, outfeed,
                    train_saver, restore, tvars)


def training_step(train, learning_rate, i, opts):
    start = time.time()
    _ = train.session.run(train.ops, feed_dict={
                          train.placeholders['learning_rate']: learning_rate})

    batch_time = (time.time() - start)
    # if not os.environ.get('TF_POPLAR_FLAGS') or '--use_synthetic_data' not in os.environ.get('TF_POPLAR_FLAGS'):
    # print(f"Task {opts['task_name']}")
    if opts['task_type'] == 'regression':
        _, _loss, per_example_loss, pred, label_ids = train.session.run(
            train.outfeed)
        pred = np.reshape(pred, (-1))
        label_ids = np.reshape(label_ids, (-1))
        pearson = sci.pearsonr(pred, label_ids)[0]
        spearman = sci.spearmanr(pred, label_ids)[0]
        loss = np.mean(_loss)

        return loss, pred, batch_time, pearson, spearman
    else:
        # 'learning_rate', 'total_loss', 'per_example_loss', 'logits', 'preds', 'acc'
        _, _loss, per_example_loss, logits, preds, acc = train.session.run(
            train.outfeed)
        loss = np.mean(_loss)
        acc = np.mean(acc)
        mean_preds = np.mean(logits)
        return loss, batch_time, acc, mean_preds


def predict_step(predict):
    start = time.perf_counter()
    _ = predict.session.run(predict.ops, feed_dict={})
    batch_time = time.perf_counter() - start
    if opts['task_type'] == 'regression':
        loss, per_example_loss, preds, label_ids = predict.session.run(
            predict.outfeed)
        # Convert to readable formats
        loss = np.mean(loss)
        preds = np.reshape(preds, (-1))
        label_ids = np.reshape(label_ids, (-1))
        pearson = sci.pearsonr(preds, label_ids)[0]
        spearman = sci.spearmanr(preds, label_ids)[0]

        tmp_output = {'loss': loss,
                      'pearson': pearson,
                      'spearman': spearman,
                      'preds': preds
                      }
    else:
        loss, per_example_loss, logits, preds, acc = predict.session.run(
            predict.outfeed)
        # Convert to readable formats
        loss = np.mean(loss)
        acc = np.mean(acc)
        tmp_output = {'loss': loss,
                      'acc': acc,
                      'preds': preds}

    return tmp_output


def main(opts):
    tf.logging.set_verbosity(tf.logging.INFO)
    """
    Set up for synthetic data.
    """
    if opts["synthetic_data"] or opts["generated_data"]:
        opts['task_name'] = 'synthetic'
        if opts['task_type'] == 'regression':
            opts['task_name'] = 'synthetic_regression'
    print(opts['task_name'])
    print(opts['task_type'])
    processors = {
        "cola": glue_data.ColaProcessor,
        "mnli": glue_data.MnliProcessor,
        "mrpc": glue_data.MrpcProcessor,
        "sst2": glue_data.Sst2Processor,
        "stsb": glue_data.StsbProcessor,
        "qqp": glue_data.QqpProcessor,
        "qnli": glue_data.QnliProcessor,
        "rte": glue_data.RteProcessor,
        "wnli": glue_data.WnliProcessor,
        "mnli-mm": glue_data.MnliMismatchProcessor,
        "ax": glue_data.AxProcessor,
        "synthetic": glue_data.SyntheticProcessor,
        "synthetic_regression": glue_data.SyntheticProcessorRegression
    }

    tokenization.validate_case_matches_checkpoint(
        do_lower_case=opts["do_lower_case"], init_checkpoint=opts["init_checkpoint"])

    tf.gfile.MakeDirs(opts["output_dir"])

    task_name = opts["task_name"].lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=opts["vocab_file"], do_lower_case=opts["do_lower_case"])
    opts["pass_in"] = (processor, label_list, tokenizer)

    train_examples = None
    # num_train_steps = None
    num_warmup_steps = None
    # So many iterations will be run for one step.
    iterations_per_step = opts['batches_per_step']
    # Avoid nan issue caused by queue length is zero.
    if opts["do_training"]:
        train_examples = processor.get_train_examples(opts["data_dir"])
        num_train_steps = int(
            len(train_examples) / opts["total_batch_size"] * opts['epochs'])
        iterations_per_epoch = len(train_examples) // opts["total_batch_size"]
        if opts.get('num_train_steps'):
            # total iterations
            iterations = opts['num_train_steps'] * opts['batches_per_step']
        else:
            iterations = iterations_per_epoch * opts['epochs']
        num_warmup_steps = int(iterations * opts["warmup"])

        tf.logging.info("***** Running training *****")
        tf.logging.info(f"  Num examples = {len(train_examples)}")
        tf.logging.info(f"  Micro batch size = {opts['micro_batch_size']}")
        tf.logging.info(f"  Num steps / epoch = {iterations_per_epoch}")
        tf.logging.info(f"  Num iterations = {iterations}")
        tf.logging.info(f"  Num steps = {num_train_steps}")
        tf.logging.info(f"  Warm steps = {num_warmup_steps}")
        tf.logging.info(f"  Warm frac = {opts['warmup']}")
        # Learning rate schedule
        lr_schedule_name = opts['lr_schedule']
        logger.info(f"Using learning rate schedule {lr_schedule_name}")
        learning_rate_schedule = make_lr_schedule(
            lr_schedule_name, opts, iterations)

    if opts["do_training"]:
        log_iterations = opts['batches_per_step'] * opts["steps_per_logs"]

        # -------------- BUILD TRAINING GRAPH ----------------
        opts['current_mode'] = 'train'
        train = build_graph(opts, iterations_per_step,
                            is_training=True)
        train.session.run(train.init)
        train.session.run(train.iterator.initializer)

        # Checkpoints load and save
        init_checkpoint_path = opts['init_checkpoint']
        if init_checkpoint_path:
            if os.path.isfile(init_checkpoint_path):
                init_checkpoint_path = os.path.splitext(
                    init_checkpoint_path)[0]

            (assignment_map, initialized_variable_names) = bert_ipu.get_assignment_map_from_checkpoint(
                train.tvars, init_checkpoint_path)

            for var in train.tvars:
                if var.name in initialized_variable_names:
                    mark = "*"
                else:
                    mark = " "
                logger.info("%-60s [%s]\t%s (%s)", var.name, mark, var.shape, var.dtype.name)

            reader = tf.train.NewCheckpointReader(init_checkpoint_path)
            load_vars = reader.get_variable_to_shape_map()

            saver_restore = tf.train.Saver(assignment_map)
            saver_restore.restore(train.session, init_checkpoint_path)

        if opts['steps_per_ckpts']:
            filepath = train.saver.save(
                train.session, opts["checkpoint_path"], global_step=0)
            logger.info(f"Saved checkpoint to {filepath}")
            ckpt_iterations = opts['batches_per_step'] * \
                opts["steps_per_ckpts"]

        else:
            i = 0

        # Tensorboard logs path
        log_path = os.path.join(opts["logs_path"], 'event')
        logger.info("Tensorboard event file path {}".format(log_path))
        summary_writer = tf.summary.FileWriter(
            log_path, train.graph, session=train.session)
        start_time = datetime.datetime.now()
        # Training loop
        if opts['task_type'] == 'regression':
            print_format = (
                "step: {step:6d}, iteration: {iteration:6d} ({percent_done:.3f}%),  epoch: {epoch:6.2f}, lr: {lr:6.4g}, loss: {loss:6.3f}, pearson: {pearson:6.3f}, spearman: {spearman:6.3f}, "
                "throughput {throughput_samples_per_sec:6.2f} samples/sec, batch time: {avg_batch_time:8.6f} s, total_time: {total_time:8.1f} s")
        else:
            print_format = (
                "step: {step:6d}, iteration: {iteration:6d} ({percent_done:.3f}%),  epoch: {epoch:6.2f}, lr: {lr:6.4g}, loss: {loss:6.3f}, acc: {acc:6.3f}, "
                "throughput {throughput_samples_per_sec:6.2f} samples/sec, batch time: {avg_batch_time:8.6f} s, total_time: {total_time:8.1f} s")
        step = 0
        start_all = time.time()
        i = 0
        total_samples = len(train_examples)

        while i < iterations:
            step += 1
            epoch = float(opts["total_batch_size"] * i) / total_samples

            learning_rate = learning_rate_schedule.get_at_step(step)

            try:
                if opts['task_type'] == 'regression':
                    loss, pred, batch_time, pearson, spearman = training_step(
                        train, learning_rate, i, opts)
                else:
                    loss, batch_time, acc, mean_preds = training_step(
                        train, learning_rate, i, opts)
            except tf.errors.OpError as e:
                raise tf.errors.ResourceExhaustedError(
                    e.node_def, e.op, e.message)

            batch_time /= iterations_per_step

            avg_batch_time = batch_time

            if i % log_iterations == 0:
                throughput = opts['total_batch_size'] / avg_batch_time

                # flush times every time it is reported
                # batch_times.clear()

                total_time = time.time() - start_all
                if opts['task_type'] == 'regression':
                    stats = OrderedDict([
                        ('step', step),
                        ('iteration', i + iterations_per_step),
                        ('percent_done', i/iterations * 100),
                        ('epoch', epoch),
                        ('lr', learning_rate),
                        ('loss', loss),
                        ('pearson', pearson),
                        ('spearman', spearman),
                        ('avg_batch_time', avg_batch_time),
                        ('throughput_samples_per_sec', throughput),
                        ('total_time', total_time),
                        ('learning_rate', learning_rate)
                    ])
                else:
                    stats = OrderedDict([
                        ('step', step),
                        ('iteration', i + iterations_per_step),
                        ('percent_done', i/iterations * 100),
                        ('epoch', epoch),
                        ('lr', learning_rate),
                        ('loss', loss),
                        ('acc', acc),
                        ('avg_batch_time', avg_batch_time),
                        ('throughput_samples_per_sec', throughput),
                        ('total_time', total_time),
                        ('learning_rate', learning_rate)
                    ])
                logger.info(print_format.format(**stats))

                train_summary = tf.Summary()
                train_summary.value.add(tag='epoch', simple_value=epoch)
                train_summary.value.add(tag='loss', simple_value=loss)
                if opts['task_type'] == 'regression':
                    train_summary.value.add(tag='pearson', simple_value=pearson)
                    train_summary.value.add(tag='spearman', simple_value=spearman)
                else:
                    train_summary.value.add(tag='acc', simple_value=acc)
                train_summary.value.add(tag='learning_rate', simple_value=learning_rate)
                train_summary.value.add(tag='througput', simple_value=throughput)

                if opts['wandb']:
                    wandb.log(dict(stats))

                summary_writer.add_summary(train_summary, step)
                summary_writer.flush()

            if i % ckpt_iterations == 0 and i > 1:
                filepath = train.saver.save(train.session, opts["checkpoint_path"],
                                            global_step=i + iterations_per_step)
                logger.info(f"Saved checkpoint to {filepath}")

            i += iterations_per_step

        # We save the final checkpoint
        finetuned_checkpoint_path = train.saver.save(train.session,
                                                     opts["checkpoint_path"],
                                                     global_step=i +
                                                     iterations_per_step)
        logger.info(f"Saved checkpoint to {finetuned_checkpoint_path}")
        train.session.close()
        end_time = datetime.datetime.now()
        consume_time = (end_time - start_time).seconds
        logger.info(f"training times: {consume_time} s")

    if opts["do_eval"]:
        eval_examples = processor.get_dev_examples(opts["data_dir"])
        num_actual_eval_examples = len(eval_examples)
        opts["eval_batch_size"] = opts['micro_batch_size'] * \
            opts['gradient_accumulation_count']

        eval_file = os.path.join(opts["output_dir"], "eval.tf_record")

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Evaluate batch size = %d", opts["eval_batch_size"])

        iterations_per_step = 1
        opts['current_mode'] = 'eval'
        predict = build_graph(opts, iterations_per_step,
                              is_training=False)
        predict.session.run(predict.init)
        predict.session.run(predict.iterator.initializer)

        if opts["init_checkpoint"] and not opts['do_training'] and opts['do_eval']:
            finetuned_checkpoint_path = opts['init_checkpoint']

        if finetuned_checkpoint_path:
            print("********** RESTORING FROM CHECKPOINT *************")
            (assignment_map, _initialized_variable_names) = bert_ipu.get_assignment_map_from_checkpoint(
                predict.tvars, finetuned_checkpoint_path)
            saver_restore = tf.train.Saver(assignment_map)
            saver_restore.restore(predict.session, finetuned_checkpoint_path)
            print("Done.")

        i = 0
        all_time_consumption = []

        iterations = int(len(
            eval_examples) // (opts['micro_batch_size'] * opts['gradient_accumulation_count']) + 1)

        all_accs = []
        all_pearson = []
        all_spearman = []
        all_loss = []
        while i < iterations:
            try:
                start = time.time()
                tmp_output = predict_step(predict)
                if opts['task_type'] == 'regression':
                    all_pearson.append(tmp_output['pearson'])
                    all_spearman.append(tmp_output['spearman'])
                else:
                    all_accs.append(tmp_output['acc'])
                all_loss.append(tmp_output['loss'])
                output_eval_file = os.path.join(
                    opts['output_dir'], "eval_results.txt")
                duration = time.time() - start
                all_time_consumption.append(
                    duration / opts["batches_per_step"])
            except tf.errors.OpError as e:
                raise tf.errors.ResourceExhaustedError(
                    e.node_def, e.op, e.message)

            i += iterations_per_step

            if len(all_loss) % 1000 == 0:
                logger.info(f"Procesing example: {len(all_loss)}")
        if opts['task_type'] == 'regression':
            tmp_output['average_pearson'] = np.mean(all_pearson)
            tmp_output['average_spearman'] = np.mean(all_spearman)
        else:
            tmp_output['average_acc'] = np.mean(all_accs)
        tmp_output['average_loss'] = np.mean(all_loss)

        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(tmp_output.keys()):
                tf.logging.info("  %s = %s", key, str(tmp_output[key]))
                writer.write("%s = %s\n" % (key, str(tmp_output[key])))
        # The time consumption of First 10 steps is not stable for time measurement.
        if len(all_time_consumption) >= 10 * 2:
            all_time_consumption = np.array(all_time_consumption[10:])
        else:
            logger.warning(
                f"if the first 10 steps is counted, the measurement of throughtput and latency is not accurate.")
            all_time_consumption = np.array(all_time_consumption)

        logger.info((
            f"inference throughput: { (opts['micro_batch_size'] * opts['gradient_accumulation_count'] ) / all_time_consumption.mean() } "
            f"exmples/sec - Latency: {all_time_consumption.mean()} {all_time_consumption.min()} "
            f"{all_time_consumption.max()} (mean min max) sec "))
        # Done evaluations

    if opts["do_predict"]:
        predict_examples = processor.get_test_examples(opts["data_dir"])
        num_actual_predict_examples = len(predict_examples)
        opts["predict_batch_size"] = opts['micro_batch_size'] * \
            opts['gradient_accumulation_count']
        tf.logging.info("***** Running prediction *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Predict batch size = %d", opts["predict_batch_size"])

        iterations_per_step = 1
        opts['current_mode'] = 'predict'
        prediction = build_graph(
            opts, iterations_per_step, is_training=False)
        prediction.session.run(prediction.init)
        prediction.session.run(prediction.iterator.initializer)

        if opts["init_checkpoint"] and not opts['do_training'] and opts['do_predict']:
            finetuned_checkpoint_path = opts['init_checkpoint']
        else:
            finetuned_checkpoint_path = False

        if finetuned_checkpoint_path:
            print("********** RESTORING FROM CHECKPOINT *************")
            (assignment_map, _initialized_variable_names) = bert_ipu.get_assignment_map_from_checkpoint(
                prediction.tvars, finetuned_checkpoint_path)
            saver_restore = tf.train.Saver(assignment_map)
            saver_restore.restore(prediction.session,
                                  finetuned_checkpoint_path)
            print("Done.")

        all_results = []
        i = 0
        all_time_consumption = []

        iterations = int(len(
            predict_examples) // (opts['micro_batch_size'] * opts['gradient_accumulation_count']) + 1)

        all_preds = []
        while i < iterations:
            try:
                start = time.time()
                tmp_output = predict_step(prediction)
                all_preds.append(tmp_output['preds'])

                output_predict_file = os.path.join(
                    opts['output_dir'], "predict_results.txt")
                duration = time.time() - start
                all_time_consumption.append(
                    duration / opts["batches_per_step"])
            except tf.errors.OpError as e:
                raise tf.errors.ResourceExhaustedError(
                    e.node_def, e.op, e.message)

            i += iterations_per_step

        all_preds = np.array(all_preds)
        all_preds = all_preds.flatten()
        headers = ["index", "prediction"]
        name_list = ["mnli", "mnli-mm", "ax", "qnli", "rte"]
        if task_name in name_list:
            all_preds = glue_data.get_output_labels(opts, all_preds)

        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results writing*****")
            for i in range(len(predict_examples)):
                if i == 0:
                    writer.write("%s\t%s\n" % (str(headers[0]), str(headers[1])))
                output_line = "%s\t%s\n" % (i, all_preds[i])
                writer.write(output_line)
        # Done predictions


def set_training_defaults(opts):
    opts['total_batch_size'] = opts['micro_batch_size'] * \
        opts['gradient_accumulation_count']
    if 'glu_reg' in [layer for stage in opts['pipeline_stages'] for layer in stage]:
        opts['task_type'] = 'regression'
    else:
        opts['task_type'] = 'classification'



def set_ipu_defaults(opts):
    opts['poplar_version'] = os.popen('popc --version').read()
    opts['hostname'] = gethostname()
    opts['datetime'] = str(datetime.datetime.now())

    if opts['seed']:
        seed = int(opts['seed'])
        random.seed(seed)
        # tensorflow seed
        tf.set_random_seed(random.randint(0, 2 ** 32 - 1))
        # numpy seed
        np.random.seed(random.randint(0, 2 ** 32 - 1))
        # ipu seed
        reset_ipu_seed(random.randint(-2**16, 2**16 - 1))


def set_defaults(opts):
    data_loader.set_defaults(opts)
    set_training_defaults(opts)
    # lr_schedule.set_defaults(opts)
    set_ipu_defaults(opts)
    log.set_defaults(opts)


def add_squad_options(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("SQuAD fine-tuning options")
    group.add_argument('--predict-file', type=str,
                       help="""SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json""")
    group.add_argument('--output-dir', type=str,
                       help="""The output directory where the model checkpoints will be written.""")
    group.add_argument('--buffer-size', type=int,
                       help="""Buffer size for training data shuffling.""")
    group.add_argument("--doc-stride", type=int, default=128,
                       help="""When splitting up a long document into chunks, how much stride to take between chunks.""")
    group.add_argument("--do-lower-case", action="store_true",
                       help="""Case sensitive or not""")
    group.add_argument("--verbose-logging", action="store_true",
                       help="""If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal SQuAD evaluation.""")
    group.add_argument("--version-2-with-negative", action="store_true",
                       help="""If true, the SQuAD examples contain some that do not have an answer.""")
    group.add_argument("--null-score-diff-threshold", type=float, default=0.0,
                       help="""If null_score - best_non_null is greater than the threshold predict null.""")
    group.add_argument("--max-query-length", type=int, default=64,
                       help="""The maximum number of tokens for the question. Questions longer than this will be truncated to this length.""")
    group.add_argument("--n-best-size", type=int, default=20,
                       help="""The total number of n-best predictions to generate in the nbest_predictions.json output file.""")
    group.add_argument("--max-answer-length", type=int, default=30,
                       help="""The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.""")
    group.add_argument("--do-predict", action="store_true",
                       help="Run inference.")
    group.add_argument("--do-training", action="store_true",
                       help="Run fine-tuning training.")
    group.add_argument("--do-eval", action="store_true",
                       help="Run GLUE evaluation script with results predicted by the inference run.")
    group.add_argument('--vocab-file', type=str,
                       help="The vocabulary file that the BERT model was trained on.")
    group.add_argument('--tfrecord-dir', type=str,
                       help="""Path to the cache directory that will contain the intermediate TFRecord datasets converted from the JSON input file.""")
    group.add_argument('--data-dir', type=str,
                       help="""The output directory where the model checkpoints will be written.""")
    group.add_argument('--task-name', type=str, default='',
                       help="""Path to the cache directory that will contain the intermediate TFRecord datasets converted from the JSON input file.""")
    return parser


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)

    opts = make_global_options([add_squad_options])

    set_defaults(opts)
    opts['distributed_worker_count'] = 1
    opts['distributed_worker_index'] = 0

    poplar_options = os.getenv('POPLAR_ENGINE_OPTIONS', 'unset')
    logger.info(f"Poplar options: {poplar_options}")
    logger.info("Command line: " + ' '.join(sys.argv))
    logger.info("Options:\n" + json.dumps(
        OrderedDict(sorted(opts.items())), indent=1))

    # Initialise Weights & Biases if available
    if opts['wandb']:
        import wandb
        wandb.init(project="tf-bert", sync_tensorboard=True,
                   name=opts['wandb_name'])
        wandb.config.update(opts)
    fine_tuned_checkpoint = None
    main(opts)
