# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import random
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.ipu import utils
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu.config import IPUConfig
from collections import namedtuple
import logging
import set_path
from din.din_model import DIN
from common.embedding import get_dataset_embed, id_embedding, get_synthetic_dataset
from common.utils import calc_auc, get_learning_rate_from_file, setup_logger
import common.log as logger

EMBEDDING_DIM = 18
TRAIN_DATA_SIZE = 1086120
VALIDATION_DATA_SIZE = 121216
STORE_LAST_MODELS_NUM = 10
use_host_eb = True
best_auc = 0.0
lowlen_global = None
highlen_global = None
seed = None

tf_log = logging.getLogger('DIN')

GraphOps = namedtuple(
    'graphOps', ['session',
                 'init',
                 'ops_val',
                 'placeholders',
                 'iterator_val',
                 'outfeed',
                 'saver'])


def graph_builder(opts, uid_embedding, mid_embedding, cat_embedding, lr,  uids, mids, cats, mid_his, cat_his, mid_mask, target, seqlen, use_negsampling=False):
    prob, loss, accuracy, train_op = DIN(uid_embedding, mid_embedding, cat_embedding, opts, False)(uids, mids, cats, mid_his, cat_his, mid_mask, seqlen, lr, target)
    return prob, loss, 0.0, accuracy, train_op


def generic_infer_graph(opts, is_training):
    data_type = 'float32'
    infer_graph = tf.Graph()
    with infer_graph.as_default():
        placeholders = {}
        placeholders["learning_rate"] = tf.compat.v1.placeholder(data_type, shape=[])
        uid_embedding, mid_embedding, cat_embedding = id_embedding(opts, is_training, seed)

        if opts['use_synthetic_data']:
            dataset_val = get_synthetic_dataset(opts)
        else:
            dataset_val = get_dataset_embed(opts, is_training=False)

        infeed_val = ipu_infeed_queue.IPUInfeedQueue(dataset_val)

        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

        with ipu_scope('/device:IPU:0'):
            def comp_fn_validate():
                def body(uids, mids, cats, mid_his, cat_his, mid_mask, target, seqlen):
                    prob, loss_total, _, accuracy, _ = graph_builder(opts, uid_embedding, mid_embedding, cat_embedding, placeholders['learning_rate'], uids, mids, cats, mid_his, cat_his, mid_mask, target, seqlen, use_negsampling=False)
                    outfeed_op = outfeed_queue.enqueue((prob, target, accuracy))
                    return outfeed_op
                return loops.repeat(opts['batches_per_step'], body, [], infeed_val)

            outputs_val = ipu_compiler.compile(comp_fn_validate, [])
            outfeed = outfeed_queue.dequeue()

        saver = tf.compat.v1.train.Saver()
        utils.move_variable_initialization_to_cpu()
        init = tf.compat.v1.global_variables_initializer()
    if opts['use_ipu_model']:
        os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"
    ipu_options = IPUConfig()
    ipu_options.optimizations.combine_embedding_lookups = True
    ipu_options.allow_recompute = True
    ipu_options.auto_select_ipus = [opts['replicas']]
    ipu_options.configure_ipu_system()
    if seed is not None:
        utils.reset_ipu_seed(seed)

    ops_val = [outputs_val]

    sess = tf.compat.v1.Session(graph=infer_graph)

    return GraphOps(sess,
                    init,
                    ops_val,
                    placeholders,
                    infeed_val,
                    outfeed,
                    saver), uid_embedding, mid_embedding, cat_embedding


def generic_init_graph(opts, is_training=True):
    graph_ops, uid_embedding, mid_embedding, cat_embedding = generic_infer_graph(opts, is_training)
    graph_ops.session.run(graph_ops.init)
    graph_ops.session.run(graph_ops.iterator_val.initializer)
    return graph_ops, uid_embedding, mid_embedding, cat_embedding


def inference(opts):
    infer_graph, uid_embedding, mid_embedding, cat_embedding = generic_init_graph(opts, False)
    path = opts['model_path']
    if path is not None and os.path.exists(path + ".meta"):
        infer_graph.saver.restore(infer_graph.session, path)
        tf_log.info(f"model {path} restored")
    else:
        tf_log.info(f"Do not restore since no model under path {path}")

    iterations = VALIDATION_DATA_SIZE * opts['epochs'] / (opts['batch_size'] * opts['replicas'])

    total_time = 0
    i = 0
    stored_arr = []
    tf_log.info(f"iterations: {iterations}")
    accs = []

    # Register the host embeddings with the session.
    with uid_embedding.register(infer_graph.session), mid_embedding.register(infer_graph.session), cat_embedding.register(infer_graph.session):
        while i < iterations:
            start = time.time()
            infer_graph.session.run(infer_graph.ops_val)
            prob, target, acc = infer_graph.session.run(infer_graph.outfeed)
            total_time = time.time() - start
            i += opts['batches_per_step']
            accuracy = np.mean(acc)
            accs.append(accuracy)
            prob_1 = prob.reshape([opts['batches_per_step']*opts['batch_size'], 2])
            prob_1 = prob_1[:, 0].tolist()
            target_1 = target.reshape([opts['batches_per_step']*opts['batch_size'], 2])
            target_1 = target_1[:, 0].tolist()
            for p, t in zip(prob_1, target_1):
                stored_arr.append([p, t])

            throughput = opts["batch_size"] * opts["batches_per_step"] / total_time
            tf_log.info(f"i={i // opts['batches_per_step']}, validation accuracy: {accuracy:.4f}, throughput:{throughput}, latency:{total_time * 1000 / opts['batches_per_step']}")
    total_time = time.time() - start
    test_auc = calc_auc(stored_arr)
    test_acc = np.mean(accs)
    tf_log.info(f"test_auc={test_auc:.4f} test_acc={test_acc:.4f}")
    infer_graph.session.close()


def add_model_arguments(parser):
    parser.add_argument("--max-seq-len", type=int, default=100, help="sequence maximum length")
    parser.add_argument("--hidden-size", type=int, default=36, help="hidden size")
    parser.add_argument("--attention-size", type=int, default=36, help="attention size")
    return parser


def add_dataset_arguments(parser):
    group = parser.add_argument_group('Dataset')
    group.add_argument('--use-synthetic-data', default=False, action='store_true', help='Use synthetic data')
    group.add_argument('--epochs', type=int, default=1, help='number of epochs')
    group.add_argument('--batches-per-step', type=int, default=160, help='Number of batches to perform on the device before returning to the host')
    return parser


def add_training_arguments(parser):
    group = parser.add_argument_group('Training')
    group.add_argument('--seed', type=int, help="set random seed")
    group.add_argument('--batch-size', type=int, default=128, help="set batch-size for training graph")
    group.add_argument('--replicas', type=int, default=1, help="Replicate graph over N workers to increase batch to batch-size*N")
    group.add_argument('--learning-rate', type=float, default=0.6, help="learning rate")
    group.add_argument('--lr-file', type=str, default='./learning_rate_schedule_file.txt', help='The learning rate file.')
    group.add_argument('--lr-type', type=str, default='fixed', choices=['fixed', 'file'], help="Choose the type of learning rate")
    group.add_argument('--model-path', type=str, default='./dnn_save_path/ckpt_noshuffDIN3', help='Place to store and restore model')
    group.add_argument('--data-type', type=str, default='float32', choices=['float32'], help='Choose the data type.')
    group.add_argument('--large-embedding', default=False, action='store_true', help="set small or large embedding size")
    group.add_argument('--use-ipu-model', default=False, action='store_true', help="use IPU model or not.")
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "CTR Model Training in Tensorflow")
    parser = add_model_arguments(parser)
    parser = add_dataset_arguments(parser)
    parser = add_training_arguments(parser)
    parser = logger.add_arguments(parser)
    args, unknown = parser.parse_known_args()
    args = vars(args)

    logger.print_setting(args)
    setup_logger(logging.INFO, tf_log)

    inference(args)
