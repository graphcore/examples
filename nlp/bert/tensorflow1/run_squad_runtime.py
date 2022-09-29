#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
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
import json
import os
import sys
import time
from collections import OrderedDict
from queue import Queue
from threading import Thread

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
from tensorflow.python.ipu import ipu_infeed_queue, ipu_outfeed_queue
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import application_compile_op
from tensorflow.python.ipu import embedded_runtime
from tensorflow.python.ipu import horovod as hvd

import gc
import log
import modeling as bert_ipu
from bert_data import data_loader
from bert_data import squad as squad_data
from bert_data import squad_results, tokenization
import ipu_utils
from log import logger
from options import make_global_options

import popdist
from poplar_options import set_poplar_engine_options
from run_squad import (
    build_network,
    should_be_pipeline_when_inference,
    build_infer_network_without_pipeline,
)
from evaluate_squad import evaluate_squad

from mpi4py import MPI
comm = MPI.COMM_WORLD


def get_exec_path(seq_length, micro_batch_size, device_mapping, pipelined):
    poplar_exec_filepath = f"./bert.poplar_exec_{str(seq_length)}_{str(micro_batch_size)}_model_per_ipu_{str(max(device_mapping) + 1 if pipelined else 1)}"
    return poplar_exec_filepath


def build_graph(opts, iterations_per_step=1, is_training=True):

    valid_graph = tf.Graph()

    with valid_graph.as_default():
        bert_config = bert_ipu.BertConfig.from_dict(
            opts, config=bert_ipu.BertConfig(vocab_size=None)
        )
        bert_config.dtype = (
            tf.float32 if opts["precision"] == "32" else tf.float16
        )

        learning_rate = None
        opts["version_2_with_negative"] = False
        valid_iterator = ipu_infeed_queue.IPUInfeedQueue(
            data_loader.load(opts, is_training=is_training)
        )
        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

        # building networks with pipeline
        if not should_be_pipeline_when_inference(opts):

            def bert_net():
                return build_infer_network_without_pipeline(
                    valid_iterator,
                    outfeed_queue,
                    iterations_per_step,
                    bert_config=bert_config,
                    opts=opts,
                )

        else:

            def bert_net():
                return build_network(
                    valid_iterator,
                    outfeed_queue,
                    iterations_per_step,
                    bert_config,
                    opts,
                    learning_rate,
                    is_training,
                )

        with ipu_scope("/device:IPU:0"):
            embedded = opts["embedded_runtime"]

            if embedded and is_training:
                raise ValueError(
                    "embedded_runtime is only to be used for inference."
                )

        exec_path = None
        compile_op = None
        poplar_exec_filepath = get_exec_path(
            opts["seq_length"],
            opts["micro_batch_size"],
            opts["device_mapping"],
            should_be_pipeline_when_inference(opts),
        )
        exec_path = os.path.join(poplar_exec_filepath)
        compile_op = (
            application_compile_op.experimental_application_compile_op(
                bert_net, output_path=exec_path, freeze_variables=True)
        )

        outfeed_queue.dequeue()

        tf.train.Saver(var_list=tf.global_variables())

        ipu.utils.move_variable_initialization_to_cpu()
        valid_init = tf.global_variables_initializer()
        tvars = tf.trainable_variables()

    with tf.Session(graph=valid_graph, config=tf.ConfigProto()) as sess:
        _ = sess.run(valid_init, [])
        # -----------------
        # Checkpoints    restore and save
        init_checkpoint_path = opts["init_checkpoint"]
        logger.info(f"At the checkpoint location {init_checkpoint_path}")
        if init_checkpoint_path:
            logger.info("Loading checkpoint...")
            if os.path.isfile(init_checkpoint_path):
                init_checkpoint_path = os.path.splitext(init_checkpoint_path)[0]
                logger.info(f"checkpoint path: {init_checkpoint_path}")

            (
                assignment_map,
                initialized_variable_names,
            ) = bert_ipu.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint_path
            )

            for var in tvars:
                if var.name in initialized_variable_names:
                    mark = "*"
                else:
                    mark = " "
                logger.info(
                    "%-60s [%s]\t%s (%s)",
                    var.name,
                    mark,
                    var.shape,
                    var.dtype.name,
                )

            reader = tf.train.NewCheckpointReader(init_checkpoint_path)
            reader.get_variable_to_shape_map()

            saver_restore = tf.train.Saver(assignment_map)
            saver_restore.restore(sess, init_checkpoint_path)
        # -----------------
        if compile_op is not None:
            logger.info(
                f"Compiling and saving Poplar executable to {poplar_exec_filepath}"
            )
            _ = sess.run(compile_op, [])
    valid_graph.finalize()


def parse_feed_dict(placeholders, data, bs, seq_length, i=0):
    """Returns the feed dict per batch with the parsed SQuAD data."""
    i = i * bs
    feed_dict = {
        placeholders[0]: data[0][i: i + bs],
        placeholders[1]: data[1][i: i + bs],
        placeholders[2]: data[2][i: i + bs],
        placeholders[3]: data[3][i: i + bs],
    }
    return feed_dict


def synthetic_feed_dict(placeholders, bs, seq_length):
    feed_dict = {
        placeholders[0]: np.zeros((bs, seq_length), dtype=np.int32),
        placeholders[1]: np.zeros((bs, seq_length), dtype=np.int32),
        placeholders[2]: np.zeros((bs, seq_length), dtype=np.int32),
        placeholders[3]: np.ones((bs,), dtype=np.int32),
    }

    return feed_dict


def run_time(opts, dataset_list=None):
    bs = opts["micro_batch_size"]
    seq_length = opts["seq_length"]
    poplar_exec_filepath = get_exec_path(
        opts["seq_length"],
        opts["micro_batch_size"],
        opts["device_mapping"],
        should_be_pipeline_when_inference(opts),
    )
    logger.info(
        f"POPLAR FILE PATH FOR EXEC: {poplar_exec_filepath}<------------------------------------------"
    )
    inputs = []
    engine_name = "my_engine"
    ctx = embedded_runtime.embedded_runtime_start(
        poplar_exec_filepath, inputs, engine_name, timeout=1000
    )
    input_ids = tf.placeholder(tf.int32, (bs, seq_length))
    input_mask = tf.placeholder(tf.int32, (bs, seq_length))
    segment_ids = tf.placeholder(tf.int32, (bs, seq_length))
    unique_ids = tf.placeholder(tf.int32, (bs,))
    placeholders = [input_ids, input_mask, segment_ids, unique_ids]
    durations = []
    master_durations = []
    durations = []
    call_result = embedded_runtime.embedded_runtime_call(placeholders, ctx)
    thread_queue = Queue()
    ipu.config.reset_ipu_configuration()
    gc.collect()
    all_results = []
    if opts["generated_data"]:
        number_of_steps = int(opts["num_iter"])
    else:
        number_of_steps = len(dataset_list[0]) // bs

    with tf.Session() as sess:
        logger.debug(f"Number of threads: {opts['num_inference_thread']}")
        logger.debug(f"Data Type: {opts['generated_data']}")
        if opts["generated_data"]:
            feed_dict = synthetic_feed_dict(
                placeholders, opts["micro_batch_size"], opts["seq_length"]
            )
        else:
            feed_dict = parse_feed_dict(
                placeholders, dataset_list, bs, seq_length, i=0
            )

        def runner(feed_dict, session):
            for step in range(number_of_steps):
                if opts["generated_data"]:
                    feed_dict = synthetic_feed_dict(
                        placeholders,
                        opts["micro_batch_size"],
                        opts["seq_length"],
                    )
                else:
                    feed_dict = parse_feed_dict(
                        placeholders, dataset_list, bs, seq_length, i=step
                    )

                start = time.time()
                ans = session.run(call_result, feed_dict=feed_dict)
                stop = time.time()
                if step % int(opts["steps_per_logs"]) == 0:
                    logger.info(
                        f"{step}/{number_of_steps}\t\t[{100*float(step/number_of_steps):.2f}%]"
                    )

                durations.append((start, stop))
                master_durations.append(stop - start)

                if opts["do_predict"]:
                    all_results.extend(
                        [squad_results.RawResult(
                            this_unique_id, this_start_logit.astype(
                                np.float64),
                            this_end_logit.astype(np.float64),)
                         for this_unique_id, this_start_logit,
                         this_end_logit in zip(ans[0],
                                               ans[1],
                                               ans[2])])
            thread_queue.put(durations, timeout=10)

        thp = [
            Thread(target=runner, args=(feed_dict, sess))
            for _ in range(opts["num_inference_thread"])
        ]
        for idx, _thread in enumerate(thp):
            _thread.start()
            logger.info(f"Thread {idx} started.")

        for idx, _thread in enumerate(thp):
            _thread.join()
            logger.info(f"Thread {idx} join.")


        durations_from_th = []
        while not thread_queue.empty():
            durations_from_th += thread_queue.get()

        latencies = [y - x for x, y in durations_from_th]
        latency = np.mean(latencies)*1000
        latency_99 = np.percentile(latencies, 99)*1000
        latency_99_9 = np.percentile(latencies, 99.9)*1000
        min_start = min([x for x, _ in durations_from_th])
        max_stop = max([y for _, y in durations_from_th])
        tput = (
            bs * opts["num_inference_thread"] *
            number_of_steps / (max_stop - min_start)
        )

        with tf.Graph().as_default(), tf.Session():
            throughputs = hvd.allgather(tf.constant([tput], name='Throughputs',
                                        dtype = tf.float32)).eval()
            all_results = hvd.allgather(tf.constant(all_results, name='InferenceResults',
                                        dtype = tf.float32)).eval()

        tput = sum(throughputs)

        print_format = ("batch size: {bs:4d}, latency avg: {latency_avg:6f} ms, latency 99p: {latency_99:6.4f} ms, latency 99p9: {latency_99_9:6.4g} ms, throughput: {throughput_samples_per_sec:6.3f} samples/sec, ")
        stats = OrderedDict(
            [
                ("bs", bs),
                ("latency_avg", latency),
                ("latency_99", latency_99),
                ("latency_99_9", latency_99_9),
                ("throughput_samples_per_sec", tput),
            ]
        )
        if not popdist.isPopdistEnvSet() or popdist.getInstanceIndex() == 0:
            logger.info(print_format.format(**stats))

    return all_results


def predict_loop(opts):
    dataset_list = None
    if not opts["generated_data"]:
        eval_examples = squad_data.read_squad_examples(
            opts["predict_file"], opts, is_training=False
        )
        tfrecord_dir = opts["tfrecord_dir"]
        if not os.path.exists(tfrecord_dir):
            os.makedirs(tfrecord_dir)

        eval_writer = squad_data.FeatureWriter(
            filename=os.path.join(tfrecord_dir, "eval.tf_record"),
            is_training=False,
        )
        eval_features = []

        tokenizer = tokenization.FullTokenizer(
            vocab_file=opts["vocab_file"], do_lower_case=opts["do_lower_case"]
        )

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
            output_fn=append_feature,
        )

        eval_writer.close()

        squad_dataset = data_loader.load(opts, is_training=False)
        squad_dataset = squad_dataset.make_one_shot_iterator()
        _input_mask_array = []
        _segment_ids_array = []
        _input_ids_array = []
        _unique_ids_array = []

        # Call `get_next()` once outside the loop to create the TensorFlow operations once.
        with tf.Session() as sess:
            next_element = squad_dataset.get_next()
            is_data = True
            while is_data:
                try:
                    output = sess.run(next_element)
                    _input_mask_array.extend(output["input_mask"])
                    _segment_ids_array.extend(output["segment_ids"])
                    _input_ids_array.extend(output["input_ids"])
                    _unique_ids_array.extend(output["unique_ids"])
                except tf.errors.OutOfRangeError:
                    print("end of training dataset")
                    is_data = False

        dataset_list = [
            _input_ids_array,
            _input_mask_array,
            _segment_ids_array,
            _unique_ids_array,
        ]

    iterations_per_step = 1

    # Calculate the number of required IPU"""
    num_ipus = (max(opts["device_mapping"]) + 1) * int(opts["replicas"])
    num_ipus = ipu_utils.next_power_of_two(num_ipus)
    # The number of acquired IPUs must be the power of 2.
    ipu_config = ipu_utils.get_config(
        fp_exceptions=opts["fp_exceptions"],
        enable_recomputation=opts["enable_recomputation"],
        disable_graph_outlining=False,
        num_required_ipus=num_ipus,
        enable_stochastic_rounding=opts["stochastic_rounding"],
        max_cross_replica_sum_buffer_size=opts[
            "max_cross_replica_sum_buffer_size"
        ],
        max_reduce_scatter_buffer_size=opts["max_reduce_scatter_buffer_size"],
        scheduler_selection="CLUSTERING",
        compile_only=False,
        ipu_id=None,
        partials_type=opts["partials_type"],
        available_memory_proportion=opts["available_memory_proportion"],
    )

    ipu_config.configure_ipu_system()

    if not popdist.isPopdistEnvSet() or popdist.getInstanceIndex() == 0:
        build_graph(
            opts, iterations_per_step, is_training=False
        )

    if popdist.isPopdistEnvSet():
        comm.barrier()

    all_results = run_time(opts, dataset_list)
    if not popdist.isPopdistEnvSet() or popdist.getInstanceIndex() == 0:
        if opts["do_predict"] is True:
            logger.info("Writing out the predictions:")
            output_dir = opts["output_dir"]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_prediction_file = os.path.join(
                output_dir, "predictions.json"
            )
            output_nbest_file = os.path.join(
                output_dir, "best_predictions.json"
            )
            output_null_log_odds_file = os.path.join(
                output_dir, "null_odds.json"
            )
            eval_features = eval_features[:num_features]
            squad_results.write_predictions(
                eval_examples,
                eval_features,
                all_results,
                opts["n_best_size"],
                opts["max_answer_length"],
                opts["do_lower_case"],
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                opts["version_2_with_negative"],
                opts["null_score_diff_threshold"],
                opts["verbose_logging"],
            )

    if opts["do_evaluation"]:
        evaluate_squad(output_prediction_file, opts)


def set_training_defaults(opts):
    opts["total_batch_size"] = (
        opts["micro_batch_size"] * opts["gradient_accumulation_count"]
    )


def set_defaults(opts):
    data_loader.set_defaults(opts)
    set_training_defaults(opts)
    log.set_defaults(opts)
    if opts["generated_data"]:
        try:
            assert opts["do_predict"] is False
            assert opts["do_evaluation"] is False
        except AssertionError:
            logger.info(
                "Cannot write predictions on synthetic generated data.\n\t\t\t\tExiting."
            )
            sys.exit(1)


def add_squad_options(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("SQuAD fine-tuning options")
    group.add_argument(
        "--predict-file",
        type=str,
        help="""SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json""",
    )
    group.add_argument(
        "--output-dir",
        type=str,
        help="""The output directory where the model checkpoints will be written.""",
    )
    group.add_argument(
        "--doc-stride",
        type=int,
        default=128,
        help="""When splitting up a long document into chunks, how much stride to take between chunks.""",
    )
    group.add_argument(
        "--do-lower-case",
        action="store_true",
        help="""Case sensitive or not""",
    )
    group.add_argument(
        "--verbose-logging",
        action="store_true",
        help="""If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal SQuAD evaluation.""",
    )
    group.add_argument(
        "--version-2-with-negative",
        action="store_true",
        help="""If true, the SQuAD examples contain some that do not have an answer.""",
    )
    group.add_argument(
        "--null-score-diff-threshold",
        type=float,
        default=0.0,
        help="""If null_score - best_non_null is greater than the threshold predict null.""",
    )
    group.add_argument(
        "--max-query-length",
        type=int,
        default=64,
        help="""The maximum number of tokens for the question. Questions longer than this will be truncated to this length.""",
    )
    group.add_argument(
        "--n-best-size",
        type=int,
        default=20,
        help="""The total number of n-best predictions to generate in the nbest_predictions.json output file.""",
    )
    group.add_argument(
        "--max-answer-length",
        type=int,
        default=30,
        help="""The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.""",
    )
    group.add_argument(
        "--do-predict", action="store_true", help="Run inference."
    )
    group.add_argument(
        "--do-training", action="store_true", help="Run fine-tuning training."
    )
    group.add_argument(
        "--do-evaluation",
        action="store_true",
        help="Run SQuAD evaluation script with results predicted by the inference run.",
    )
    group.add_argument(
        "--vocab-file",
        type=str,
        help="The vocabulary file that the BERT model was trained on.",
    )
    group.add_argument(
        "--tfrecord-dir",
        type=str,
        help="""Path to the cache directory that will contain the intermediate TFRecord datasets converted from the JSON input file.""",
    )
    group.add_argument(
        "--num-synthetic-dataset-samples",
        type=int,
        default=1000000,
        help="""Number of samples generated in the synthetic dataset""",
    )
    # Inference
    group.add_argument(
        "--embedded-runtime",
        action="store_true",
        default=False,
        help="Runs model on the embedded runtime.",
    )
    group.add_argument(
        "--dump-raw-timings",
        action="store_true",
        default=False,
        help="Dumps out elapsed times per execution to timings.txt",
    )
    group.add_argument(
        "--num-iter",
        action="store",
        default=1000,
        help="Number of iterations to run inference for.",
    )
    group.add_argument(
        "--num-inference-thread",
        action="store",
        default=2,
        help="Number of threads to use.",
    )
    return parser


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)

    opts = make_global_options([add_squad_options])

    set_defaults(opts)
    opts["num_inference_thread"] = int(opts["num_inference_thread"])
    poplar_options = os.getenv("POPLAR_ENGINE_OPTIONS", "unset")

    logger.info(f"Poplar options: {poplar_options}")
    logger.info("Command line: " + " ".join(sys.argv))
    logger.info(
        "Options:\n" + json.dumps(OrderedDict(sorted(opts.items())), indent=1)
    )

    set_poplar_engine_options(
        execution_profile=opts["execution_profile"],
        memory_profile=opts["memory_profile"],
        profile_dir=str(opts["profile_dir"]),
        sync_replicas_independently=opts["replicas"] > 1 and opts["sync_replicas_independently"],
        synthetic_data=opts["synthetic_data"],
        tensorflow_progress_bar=opts["progress_bar"],
        ipu_replica_identical_seed=opts["ipu_replica_identical_seed"],
    )

    hvd.init()

    predict_loop(opts)
