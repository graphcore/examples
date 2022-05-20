# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Traing CTR Model on Graphcore IPUs.
"""

import os
import time
import random
import argparse
import numpy as np
import logging
import tensorflow.compat.v1 as tf
from collections import namedtuple
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import utils
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu.config import IPUConfig
import set_path
from common.utils import calc_auc, setup_logger
from common.embedding import get_dataset_embed, id_embedding, get_synthetic_dataset
import common.log as logger
from dien.dien_model import DIEN


VALIDATION_DATA_SIZE = 121216
tf_log = logging.getLogger('DIEN')


GraphOps = namedtuple(
    'graphOps', ['graph',
                 'session',
                 'init',
                 'ops',
                 'placeholders',
                 'iterator',
                 'outfeed',
                 'saver'])


def get_tf_datatype(opts):
    dtypes = opts["precision"].split('.')
    master_dtype = tf.float16 if dtypes[1] == '16' else tf.float32
    return master_dtype


def graph_builder(opts, uid_embedding, mid_embedding, cat_embedding, lr,
                  uids, mids, cats, mid_his, cat_his, mid_mask, target, sl,
                  use_negsampling=True):
    master_dtype = get_tf_datatype(opts)
    return DIEN(opts, uid_embedding, mid_embedding, cat_embedding, master_dtype)(uids, mids, cats, mid_his, cat_his, mid_mask, sl, None, None, lr, target)


def generic_graph(opts, is_training):
    master_dtype = get_tf_datatype(opts)
    graph = tf.Graph()

    with graph.as_default():
        placeholders = {}
        placeholders["learning_rate"] = tf.placeholder(master_dtype, shape=[])
        uid_embedding, mid_embedding, cat_embedding = id_embedding(opts, is_training, opts['seed'])
        if opts['use_synthetic_data']:
            dataset = get_synthetic_dataset(opts)
        else:
            dataset = get_dataset_embed(opts, False)
        infeed = ipu_infeed_queue.IPUInfeedQueue(dataset)
        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

        with ipu_scope('/device:IPU:0'):
            def comp_fn():
                def body(uids, mids, cats, mid_his, cat_his, mid_mask, target, sl):
                    prob, accuracy = graph_builder(opts, uid_embedding, mid_embedding, cat_embedding, placeholders['learning_rate'], uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, use_negsampling=False)
                    with tf.control_dependencies([prob]):
                        return outfeed_queue.enqueue((prob, target, accuracy))
                return loops.repeat(opts['batches_per_step'], body, [], infeed)

            outputs = ipu_compiler.compile(comp_fn, [])
            outfeed = outfeed_queue.dequeue()

        saver = tf.train.Saver()

        utils.move_variable_initialization_to_cpu()
        init = tf.global_variables_initializer()
        if opts['use_ipu_model']:
            os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    ipu_options = IPUConfig()
    ipu_options.allow_recompute = True
    ipu_options.auto_select_ipus = [opts['replicas']]
    ipu_options.optimizations.maximum_cross_replica_sum_buffer_size = 10000000
    ipu_options.optimizations.maximum_inter_ipu_copies_buffer_size = 10000000
    ipu_options.configure_ipu_system()

    graph_outputs = [outputs]

    sess = tf.Session(graph=graph)

    return GraphOps(graph,
                    sess,
                    init,
                    graph_outputs,
                    placeholders,
                    infeed,
                    outfeed,
                    saver), uid_embedding, mid_embedding, cat_embedding


def inference(opts):
    infer, uid_embedding, mid_embedding, cat_embedding = generic_graph(opts, False)
    infer.session.run(infer.init)
    infer.session.run(infer.iterator.initializer)

    path = opts['model_path']
    if path is not None and os.path.exists(path+".meta"):
        infer.saver.restore(infer.session, path)
        tf_log.debug(f"model {path} restored")
    else:
        tf_log.debug(f"Do not restore since no model under path {path}")


    steps = VALIDATION_DATA_SIZE * opts['epochs'] / opts['micro_batch_size'] / opts["batches_per_step"]

    i = 0
    stored_arr = []
    tf_log.debug(f"steps: {steps}")
    accs = []
    total_time = 0
    with uid_embedding.register(infer.session), mid_embedding.register(infer.session), cat_embedding.register(infer.session):
        while i < steps:
            start = time.time()
            infer.session.run(infer.ops)
            prob, target, acc = infer.session.run(infer.outfeed)
            time_one_iteration = time.time() - start
            if i > 0:
                total_time = total_time + time_one_iteration
            i += 1
            accuracy = np.mean(acc)
            accs.append(accuracy)
            prob_1 = prob.reshape([opts['batches_per_step']*opts['micro_batch_size'], 2*opts['replicas']])
            prob_1 = prob_1[:, 0].tolist()
            target_1 = target.reshape([opts['batches_per_step']*opts['micro_batch_size'], 2*opts['replicas']])
            target_1 = target_1[:, 0].tolist()
            for p, t in zip(prob_1, target_1):
                stored_arr.append([p, t])
            throughput = opts["micro_batch_size"] * opts["batches_per_step"] / time_one_iteration
            tf_log.info(f"i={i // opts['batches_per_step']},validation accuracy: {accuracy}, throughput:{throughput}, latency:{time_one_iteration * 1000 / opts['batches_per_step']}")
    test_auc = calc_auc(stored_arr)
    test_acc = np.mean(accs)
    tf_log.info(f"test_auc={test_auc:.4f} test_acc={test_acc:.4f}")

    infer.session.close()
    if steps > 1:
        total_recomm_num = opts["micro_batch_size"] * (i - 1) * opts["batches_per_step"]
        throughput = float(total_recomm_num) / float(total_time)
        latency = float(total_time) * 1000 / float((i - 1) * opts["batches_per_step"])
        tf_log.info(f"Total recommendations: {total_recomm_num:d}")
        tf_log.info(f"Process time in seconds is {total_time:.3f}")
        tf_log.info(f"recommendations/second is {throughput:.3f}")
        tf_log.info(f"latency in miliseconds is {latency:.3f}")


def add_model_arguments(parser):
    parser.add_argument("--max-seq-len", type=int, default=100, help="sequence maximum length")
    parser.add_argument("--hidden-size", type=int, default=36, help="hidden size")
    parser.add_argument("--attention-size", type=int, default=36, help="attention size")
    parser.add_argument("--precision", type=str, default="32.32", choices=["32.32"], help="Setting of Ops and Master datatypes")
    parser.add_argument("--gru-type", type=str, default="PopnnGRU", choices=["TfnnGRU", "PopnnGRU"], help="choose GRU")
    parser.add_argument("--augru-type", type=str, default="PopnnAUGRU", choices=["TfAUGRU", "PopnnAUGRU"], help="choose AUGRU")
    return parser


def add_dataset_arguments(parser):
    group = parser.add_argument_group('Dataset')
    group.add_argument('--use-synthetic-data', default=False, action='store_true', help='Use synthetic data')
    group.add_argument('--epochs', type=float, default=1, help='number of epochs')
    group.add_argument('--batches-per-step', type=int, default=1600, help='Number of batches to perform on the device before returning to the host')
    return parser


def add_training_arguments(parser):
    group = parser.add_argument_group('Training')
    group.add_argument('--seed', type=int, default=3, help = "set random seed")
    group.add_argument('--micro-batch-size', type=int, default=128, help = "set micro-batch-size for training graph")
    group.add_argument('--large-embedding', default=False, action='store_true', help="set small or large embedding size")
    group.add_argument('--replicas', type=int, default=1, help = "Replicate graph over N workers to increase batch to batch-size*N")
    group.add_argument('--model-path', type=str, default='./dnn_save_path/ckpt_noshuffDIEN3', help='Place to store and restore model')
    group.add_argument('--use-ipu-model', default=False, action='store_true', help="use IPU model or not.")
    group.add_argument('--use-ipu-emb', default=False, action='store_true', help = "Use host embeddig or put embedding on ipu.")
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "CTR Model Training in TensorFlow", add_help = False)
    parser = add_model_arguments(parser)
    parser = add_dataset_arguments(parser)
    parser = add_training_arguments(parser)
    parser = logger.add_arguments(parser)
    args, _ = parser.parse_known_args()
    args = vars(args)
    logger.print_setting(args, is_dien=False, is_training=False)
    setup_logger(logging.DEBUG, tf_log, name='dien_log.txt')

    inference(args)
