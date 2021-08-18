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
import collections
import datetime
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import OrderedDict, deque, namedtuple
from socket import gethostname

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
from tensorflow.python.ipu import ipu_infeed_queue, ipu_outfeed_queue
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu.utils import reset_ipu_seed

import log
import modeling as bert_ipu
from bert_data import data_loader
from bert_data import squad as squad_data
from bert_data import squad_results, tokenization
from ipu_optimizer import get_optimizer
from ipu_utils import get_config, stages_constructor
from log import logger
from lr_schedules import make_lr_schedule
from options import make_global_options

from typing import List
from itertools import chain


# Paths to external SQuAD files
SQUADV11_TRUTH_PATH = 'data/squad/dev-v1.1.json'
SQUADV11_EVAL_SCRIPT_PATH = 'data/squad/evaluate-v1.1.py'

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
    assert opts['gradient_accumulation_count'] % (len(opts['pipeline_stages'])*2) == 0

    layers = {
        'emb': model.embedding_lookup_layer,
        'pos': model.embedding_postprocessor_layer,
        'hid': model.encoder,
        'loc': model.get_loc_logic_output_layer
    }
    stage_layer_list = []
    for stage in opts['pipeline_stages']:
        func_list = []
        for layer in stage:
            # embedding layer and mlm layer can be splited to mutliple IPUs, so need to be dealt with separately
            func_list.append(layers[layer])
        stage_layer_list.append(func_list)

    if is_training:
        computational_stages = stages_constructor(
            stage_layer_list, ['learning_rate'], ['learning_rate', 'total_loss'])
    else:
        computational_stages = stages_constructor(
            stage_layer_list, [], ['unique_ids', 'start_logits', 'end_logits'])

    return computational_stages


def build_network(infeed,
                  outfeed,
                  iterations_per_step=1,
                  bert_config=None,
                  opts=None,
                  learning_rate=None,
                  is_training=True):
    # build model
    pipeline_model = bert_ipu.BertModel(bert_config,
                                        is_training=is_training)

    # build stages & device mapping
    computational_stages = build_squad_pipeline_stages(
        pipeline_model, bert_config, opts, is_training)
    device_mapping = opts['device_mapping']
    logger.info(f"************* computational stages: *************\n{computational_stages}")
    logger.info(f"************* device mapping: *************\n{device_mapping}")

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
        def optimizer_function(learning_rate, total_loss):
            optimizer = get_optimizer(learning_rate, opts['loss_scaling'], opts['replicas'], opts)
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


def should_be_pipeline_when_inference(opts: dict):
    return not (len(set(opts["device_mapping"])) == 1 and
                opts["do_training"] is False and opts["do_predict"] is True)


def merge_compute_stages(model, opts: dict):
    """merge computation stages into one single model

    Args:
        stages (List[List]): the list of list of model blocks
    """
    pipeline_stage_backup = opts["pipeline_stages"]
    opts["pipeline_stages"] = [[i for i in chain(*pipeline_stage_backup)]]
    computational_stages = build_squad_infer_stages(model, opts)
    opts["pipeline_stages"] = pipeline_stage_backup

    return computational_stages


def build_squad_infer_stages(model, opts):
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

    layers = {
        'emb': model.embedding_lookup_layer,
        'pos': model.embedding_postprocessor_layer,
        'hid': model.encoder,
        'loc': model.get_loc_logic_output_layer
    }
    stage_layer_list = []
    for stage in opts['pipeline_stages']:
        func_list = []
        for layer in stage:
            # embedding layer and mlm layer can be splited to mutliple IPUs, so need to be dealt with separately
            func_list.append(layers[layer])
        stage_layer_list.append(func_list)

    computational_stages = stages_constructor(
        stage_layer_list, ["unique_ids", "input_ids", "input_mask",
                           "segment_ids", "start_positions", "end_positions"],
                          ['unique_ids', 'start_logits', 'end_logits'])

    return computational_stages


def build_network_without_pipeline(computational_stages,
                                   infeed,
                                   outfeed,
                                   opts=None,
                                   iterations_per_step=1):
    """return loop.repeat object"""

    def inference_graph(unique_ids=None,
                        input_ids=None,
                        input_mask=None,
                        segment_ids=None,
                        start_positions=None,
                        end_positions=None):
        o = computational_stages[0](unique_ids = unique_ids,
                                    input_ids = input_ids,
                                    input_mask = input_mask,
                                    segment_ids = segment_ids,
                                    start_positions = start_positions,
                                    end_positions = end_positions,)
        o = outfeed.enqueue(o)
        return o

    r = ipu.loops.repeat(iterations_per_step, inference_graph, inputs=[], infeed_queue=infeed)
    return r


def build_infer_network_without_pipeline(infeed,
                                         outfeed,
                                         iterations_per_step=1,
                                         bert_config=None,
                                         opts=None):
    # build model
    model = bert_ipu.BertModel(bert_config, is_training=False)

    # build stages & device mapping
    computational_stages = merge_compute_stages(model, opts)
    return build_network_without_pipeline(computational_stages, infeed, outfeed, opts=opts,
                                          iterations_per_step = iterations_per_step)


def build_graph(opts, iterations_per_step=1, is_training=True, feed_name=None):

    train_graph = tf.Graph()
    with train_graph.as_default():
        bert_config = bert_ipu.BertConfig.from_dict(opts)
        bert_config.dtype = tf.float32 if opts["precision"] == '32' else tf.float16
        placeholders = dict()

        if is_training:
            placeholders['learning_rate'] = tf.placeholder(bert_config.dtype, shape=[])
            learning_rate = placeholders['learning_rate']
        else:
            learning_rate = None

        train_iterator = ipu_infeed_queue.IPUInfeedQueue(data_loader.load(opts, is_training=is_training))
        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

        # building networks with pipeline
        if not should_be_pipeline_when_inference(opts):
            def bert_net():
                return build_infer_network_without_pipeline(train_iterator,
                                                            outfeed_queue,
                                                            iterations_per_step,
                                                            bert_config=bert_config,
                                                            opts=opts)
        else:
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
    if should_be_pipeline_when_inference(opts):
        ipu_config = get_config(fp_exceptions=opts["fp_exceptions"],
                                enable_recomputation=opts["enable_recomputation"],
                                disable_graph_outlining=False,
                                num_required_ipus=num_ipus,
                                enable_stochastic_rounding=opts['stochastic_rounding'],
                                max_cross_replica_sum_buffer_size=opts['max_cross_replica_sum_buffer_size'],
                                scheduler_selection='CLUSTERING',
                                compile_only=False,
                                ipu_id=None)
    else:
        ipu_config = get_config(fp_exceptions=opts["fp_exceptions"],
                                enable_recomputation=opts["enable_recomputation"],
                                disable_graph_outlining=False,
                                num_required_ipus=num_ipus,
                                enable_stochastic_rounding=opts['stochastic_rounding'],
                                max_cross_replica_sum_buffer_size=opts['max_cross_replica_sum_buffer_size'],
                                scheduler_selection='CLUSTERING',
                                compile_only=False,
                                ipu_id=None,
                                available_memory_proportion=opts["available_memory_proportion"],
                                partials_type=opts["partials_type"])

    ipu_config.configure_ipu_system()

    train_sess = tf.Session(graph=train_graph)

    return GraphOps(train_graph, train_sess, train_init, [train], placeholders, train_iterator, outfeed,
                    train_saver, restore, tvars)


def training_step(train, learning_rate):
    start = time.time()
    _ = train.session.run(train.ops, feed_dict={
                          train.placeholders['learning_rate']: learning_rate})
    batch_time = (time.time() - start)
    if not os.environ.get('TF_POPLAR_FLAGS') or '--use_synthetic_data' not in os.environ.get('TF_POPLAR_FLAGS'):
        _, _loss = train.session.run(train.outfeed)
        loss = np.mean(_loss)
    else:
        loss = 0
    return loss, batch_time


def predict_step(predict):
    start = time.perf_counter()
    _ = predict.session.run(predict.ops, feed_dict={})
    batch_time = time.perf_counter() - start
    _unique_ids, _start_logits, _end_logits = predict.session.run(predict.outfeed)
    return _unique_ids, _start_logits, _end_logits, batch_time


def training_loop(opts):
    consume_time = None
    train_examples = squad_data.read_squad_examples(opts['train_file'], opts, is_training=True)
    total_samples = len(train_examples)
    logger.info(f"Total samples {total_samples}")
    iterations_per_epoch = total_samples // opts["total_batch_size"]
    log_iterations = opts['batches_per_step'] * opts["steps_per_logs"]
    ckpt_iterations = opts['batches_per_step'] * opts["steps_per_ckpts"]

    if opts.get('num_train_steps'):
        # total iterations
        iterations = opts['num_train_steps'] * opts['batches_per_step']
    elif opts.get('epochs'):
        iterations = iterations_per_epoch * opts['epochs']
    else:
        logger.error("One between epochs and num_train_step must be set")
        sys.exit(os.EX_OK)

    logger.info(f"Training will last {iterations} iterations and {iterations//opts['batches_per_step']} steps will be executed.")

    # So many iterations will be run for one step.
    iterations_per_step = opts['batches_per_step']
    # Avoid nan issue caused by queue length is zero.
    queue_len = iterations_per_epoch // iterations_per_step
    if queue_len == 0:
        queue_len = 1
    batch_times = deque(maxlen=queue_len)

    total_steps = (iterations // opts['batches_per_step'])*opts['batches_per_step']

    # Learning rate schedule
    lr_schedule_name = opts['lr_schedule']
    logger.info(f"Using learning rate schedule {lr_schedule_name}")
    learning_rate_schedule = make_lr_schedule(lr_schedule_name, opts, total_steps)

    # -------------- BUILD TRAINING GRAPH ----------------
    train = build_graph(opts, iterations_per_step,
                        is_training=True, feed_name="trainfeed")
    train.session.run(train.init)
    train.session.run(train.iterator.initializer)

    # Checkpoints restore and save
    init_checkpoint_path = opts['init_checkpoint']
    if init_checkpoint_path and not opts.get('generated_data', False):
        if os.path.isfile(init_checkpoint_path):
            init_checkpoint_path = os.path.splitext(init_checkpoint_path)[0]

        (assignment_map, initialized_variable_names) = bert_ipu.get_assignment_map_from_checkpoint(train.tvars, init_checkpoint_path)

        reader = tf.train.NewCheckpointReader(init_checkpoint_path)
        load_vars = reader.get_variable_to_shape_map()

        saver_restore = tf.train.Saver(assignment_map)
        saver_restore.restore(train.session, init_checkpoint_path)

    if opts['steps_per_ckpts']:
        filepath = train.saver.save(
            train.session, opts["checkpoint_path"], global_step=0)
        logger.info(f"Saved checkpoint to {filepath}")

    if opts.get('restore_dir'):
        restore_path = opts['restore_dir']
        if os.path.isfile(restore_path):
            latest_checkpoint = os.path.splitext(restore_path)[0]
        else:
            latest_checkpoint = tf.train.latest_checkpoint(restore_path)
        ckpt_pattern = re.compile(".*ckpt-([0-9]+)$")
        i = int(ckpt_pattern.match(latest_checkpoint).groups()[0]) + 1
        train.saver.restore(train.session, latest_checkpoint)
        epoch = float(opts["total_batch_size"] * (i + iterations_per_step)) / total_samples
    else:
        i = 0

    # Tensorboard logs path
    log_path = os.path.join(opts["logs_path"], 'event')
    logger.info("Tensorboard event file path {}".format(log_path))
    summary_writer = tf.summary.FileWriter(log_path, train.graph, session=train.session)
    start_time = datetime.datetime.now()

    # Training loop
    print_format = (
        "step: {step:6d}, iteration: {iteration:6d}, epoch: {epoch:6.2f}, lr: {lr:6.4g}, loss: {loss:6.3f}, "
        "throughput {throughput_samples_per_sec:6.2f} samples/sec, batch time: {avg_batch_time:8.6f} s, total_time: {total_time:8.1f} s")
    step = 0
    start_all = time.time()

    while i < iterations:
        step += 1
        epoch = float(opts["total_batch_size"] * i) / total_samples

        learning_rate = learning_rate_schedule.get_at_step(step)

        try:
            loss, batch_time = training_step(
                train, learning_rate)
        except tf.errors.OpError as e:
            raise tf.errors.ResourceExhaustedError(
                e.node_def, e.op, e.message)

        batch_time /= iterations_per_step

        if i != 0:
            batch_times.append([batch_time])
            avg_batch_time = np.mean(batch_times)
        else:
            avg_batch_time = batch_time

        if i % log_iterations == 0:
            throughput = opts['total_batch_size'] / avg_batch_time

            # flush times every time it is reported
            batch_times.clear()

            total_time = time.time() - start_all

            stats = OrderedDict([
                ('step', step),
                ('iteration', i + iterations_per_step),
                ('epoch', epoch),
                ('lr', learning_rate),
                ('loss', loss),
                ('avg_batch_time', avg_batch_time),
                ('throughput_samples_per_sec', throughput),
                ('total_time', total_time),
                ('learning_rate', learning_rate)
            ])
            logger.info(print_format.format(**stats))

            train_summary = tf.Summary()
            train_summary.value.add(tag='epoch', simple_value=epoch)
            train_summary.value.add(tag='loss', simple_value=loss)
            train_summary.value.add(tag='learning_rate', simple_value=learning_rate)
            train_summary.value.add(tag='througput', simple_value=throughput)

            if opts['wandb']:
                wandb.log(dict(stats))

            summary_writer.add_summary(train_summary, step)
            summary_writer.flush()

        if i % ckpt_iterations == 0:
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
    return finetuned_checkpoint_path


def predict_loop(opts, finetuned_checkpoint_path=None):
    i = 0
    eval_examples = squad_data.read_squad_examples(opts["predict_file"], opts, is_training=False)

    tfrecord_dir = opts['tfrecord_dir']
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)

    eval_writer = squad_data.FeatureWriter(
        filename=os.path.join(tfrecord_dir, "eval.tf_record"),
        is_training=False)
    eval_features = []

    tokenizer = tokenization.FullTokenizer(
        vocab_file=opts['vocab_file'], do_lower_case=opts['do_lower_case'])

    def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)

    # Create eval.tfrecord
    num_features = squad_data.convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=opts["seq_length"],
        doc_stride=opts["doc_stride"],
        max_query_length=opts["max_query_length"],
        is_training=False,
        output_fn=append_feature)

    eval_writer.close()
    iterations_per_step = 1
    predict = build_graph(opts, iterations_per_step, is_training=False, feed_name="evalfeed")
    predict.session.run(predict.init)
    predict.session.run(predict.iterator.initializer)

    if opts["init_checkpoint"] and not finetuned_checkpoint_path:
        finetuned_checkpoint_path = opts['init_checkpoint']

    # Note that finetuned_checkpoint_path could be already set during "do_predict"
    if finetuned_checkpoint_path and not opts.get('generated_data', False):
        (assignment_map, _initialized_variable_names) = bert_ipu.get_assignment_map_from_checkpoint(
            predict.tvars, finetuned_checkpoint_path)
        saver_restore = tf.train.Saver(assignment_map)
        saver_restore.restore(predict.session, finetuned_checkpoint_path)
        assert len(assignment_map) >= 127

    all_results = []
    iterations = len(eval_features) // (opts['batch_size'] * opts['gradient_accumulation_count']) + 1

    all_time_consumption = []
    while i < iterations:
        try:
            # start = time.time()
            unique_ids, start_logits, end_logits, batch_duration = predict_step(predict)
            # duration = time.time() - start
            # all_time_consumption.append(duration)
            all_time_consumption.append(batch_duration / opts["batches_per_step"])
        except tf.errors.OpError as e:
            raise tf.errors.ResourceExhaustedError(
                e.node_def, e.op, e.message)

        i += iterations_per_step

        if len(all_results) % 1000 == 0:
            logger.info(f"Procesing example: {len(all_results)}")

        # The outfeed shape is [batches_per_step, num_replicas (if replication enabled), micro_batch_size, seq_len].
        # Flatten to keep only the last dimension
        num_samples = np.prod(unique_ids.shape)
        seq_len = opts['seq_length']
        unique_ids = unique_ids.reshape([num_samples])
        start_logits = start_logits.reshape([num_samples, seq_len])
        end_logits = end_logits.reshape([num_samples, seq_len])

        for j in range(num_samples):
            unique_id = unique_ids[j]
            start_logit = start_logits[j, :].tolist()
            end_logit = end_logits[j, :].tolist()
            all_results.append(
                squad_results.RawResult(
                    unique_id=unique_id,
                    start_logits=start_logit,
                    end_logits=end_logit))

    if len(all_time_consumption) >= 10 * 2:    # The time consumption of First 10 steps is not stable for time measurement.
        all_time_consumption = np.array(all_time_consumption[10:])
    else:
        logger.warning(f"if the first 10 steps is counted, the measurement of throughtput and latency is not accurate.")
        all_time_consumption = np.array(all_time_consumption)

    logger.info((
        f"inference throughput: { (opts['batch_size'] * opts['gradient_accumulation_count'] if should_be_pipeline_when_inference(opts) else opts['batch_size']) / all_time_consumption.mean() } "
        f"exmples/sec - Latency: {all_time_consumption.mean()} {all_time_consumption.min()} "
        f"{all_time_consumption.max()} (mean min max) sec "))
    # Done predictions

    output_dir = opts['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_prediction_file = os.path.join(output_dir, "predictions.json")
    output_nbest_file = os.path.join(output_dir, "best_predictions.json")
    output_null_log_odds_file = os.path.join(output_dir, "null_odds.json")
    eval_features = eval_features[:num_features]
    squad_results.write_predictions(eval_examples, eval_features, all_results, opts["n_best_size"],
                                    opts["max_answer_length"], opts["do_lower_case"], output_prediction_file,
                                    output_nbest_file, output_null_log_odds_file, opts["version_2_with_negative"],
                                    opts["null_score_diff_threshold"], opts["verbose_logging"])

    predict.session.close()

    if opts['do_evaluation']:
        stdout = subprocess.check_output(['python3', SQUADV11_EVAL_SCRIPT_PATH, SQUADV11_TRUTH_PATH, output_prediction_file])
        em_f1_results = json.loads(stdout)
        logger.info(f"Evaluation results: Exact Match: {em_f1_results['exact_match']:5.2f}, F1: {em_f1_results['f1']:5.2f}")
        if opts['wandb']:
            for k, v in em_f1_results.items():
                wandb.run.summary[k] = v


def set_training_defaults(opts):
    opts['total_batch_size'] = opts['batch_size'] * opts['gradient_accumulation_count']


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
                       help= """The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.""")
    group.add_argument("--do-predict", action="store_true",
                       help="Run inference.")
    group.add_argument("--do-training", action="store_true",
                       help="Run fine-tuning training.")
    group.add_argument("--do-evaluation", action="store_true",
                       help="Run SQuAD evaluation script with results predicted by the inference run.")
    group.add_argument('--vocab-file', type=str,
                       help="The vocabulary file that the BERT model was trained on.")
    group.add_argument('--tfrecord-dir', type=str,
                       help= """Path to the cache directory that will contain the intermediate TFRecord datasets converted from the JSON input file.""")
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
        wandb.init(project="tf-bert", sync_tensorboard=False, name=opts['wandb_name'])
        wandb.config.update(opts)

    fine_tuned_checkpoint = None
    if opts['do_training']:
        fine_tuned_checkpoint = training_loop(opts)

    if opts['do_predict']:
        predict_loop(opts, fine_tuned_checkpoint)
