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

import time
import random
import argparse
import numpy as np
import tensorflow as tf
from collections import namedtuple
import logging
import set_path
from common.embedding import get_dataset_embed, id_embedding, get_synthetic_dataset
from tensorflow.python.ipu import utils
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu.config import IPUConfig

from din.din_model import DIN
from common.utils import get_learning_rate_from_file, setup_logger
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
                 'ops_train',
                 'placeholders',
                 'iterator_train',
                 'outfeed',
                 'saver'])


def graph_builder(opts, uid_embedding, mid_embedding, cat_embedding, lr,  uids, mids, cats, mid_his, cat_his, mid_mask, target, seqlen, use_negsampling=False):

    prob, loss, accuracy, train_op = DIN(uid_embedding, mid_embedding, cat_embedding, opts, True, seed)(uids, mids, cats, mid_his, cat_his, mid_mask, seqlen, lr, target)
    return prob, loss, 0.0, accuracy, train_op


def generic_train_graph(opts, is_training):
    data_type = 'float32'
    train_graph = tf.Graph()
    with train_graph.as_default():
        placeholders = {}
        placeholders["learning_rate"] = tf.compat.v1.placeholder(data_type, shape=[])
        uid_embedding, mid_embedding, cat_embedding = id_embedding(opts, is_training, seed)

        if opts['use_synthetic_data']:
            dataset_train = get_synthetic_dataset(opts)
        else:
            dataset_train = get_dataset_embed(opts, is_training=True)

        infeed_train = ipu_infeed_queue.IPUInfeedQueue(dataset_train)

        with ipu_scope('/device:IPU:0'):
            def comp_fn():
                def body(total_loss, total_aux_loss, total_accuracy, uids, mids, cats, mid_his, cat_his, mid_mask, target, seqlen):
                    prob, loss, aux_loss, accuracy, grad_op = graph_builder(opts, uid_embedding, mid_embedding, cat_embedding, placeholders['learning_rate'], uids, mids, cats, mid_his, cat_his, mid_mask, target, seqlen, use_negsampling=False)

                    with tf.control_dependencies([grad_op]):
                        return total_loss + loss, total_aux_loss + aux_loss, total_accuracy + accuracy

                return loops.repeat(opts['batches_per_step'], body, [tf.constant(0, getattr(np, 'float32'))] * 3, infeed_train)

            outputs_train = ipu_compiler.compile(comp_fn, [])
            avg_loss, avg_aux_loss, avg_accuracy = [x / opts['batches_per_step'] for x in outputs_train]
            outfeed = None

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

    ops_train = [avg_loss, avg_aux_loss, avg_accuracy]
    sess = tf.compat.v1.Session(graph=train_graph)

    return GraphOps(sess,
                    init,
                    ops_train,
                    placeholders,
                    infeed_train,
                    outfeed,
                    saver), uid_embedding, mid_embedding, cat_embedding


def generic_init_graph(opts, is_training=True):
    graph_ops, uid_embedding, mid_embedding, cat_embedding = generic_train_graph(opts, is_training)
    graph_ops.session.run(graph_ops.init)
    graph_ops.session.run(graph_ops.iterator_train.initializer)
    return graph_ops, uid_embedding, mid_embedding, cat_embedding


def get_learning_rate(opts, i):
    if opts['lr_type'] == 'file':
        lr_value = get_learning_rate_from_file(opts['lr_file'], i)
    else:
        lr_value = opts['learning_rate']
    return lr_value


def train_process(opts, restore=False):
    path = opts['model_path']
    train_graph, uid_embedding, mid_embedding, cat_embedding = generic_init_graph(opts)

    if restore is True:
        train_graph.saver.restore(train_graph.session, path)
        tf_log.info(f"model {path} restored")

    epochs = opts['epochs']
    iterations_per_epoch = TRAIN_DATA_SIZE / (opts['micro_batch_size'] * opts['replicas'])
    iterations = epochs * iterations_per_epoch
    steps = iterations // opts["batches_per_step"]

    total_time = 0
    i = 0
    tf_log.info(f"iterations is {iterations}")
    tf_log.info(f"Steps={steps}")

    # Register the host embeddings with the session.
    with uid_embedding.register(train_graph.session), mid_embedding.register(train_graph.session), cat_embedding.register(train_graph.session):
        while i < steps:
            epoch = int(i * opts["batches_per_step"] // iterations_per_epoch) + 1
            lr_value = get_learning_rate(opts, i)
            start = time.time()
            loss, _, accuracy = train_graph.session.run(train_graph.ops_train, feed_dict = {train_graph.placeholders["learning_rate"]: lr_value})
            avg_time = time.time() - start
            total_time += avg_time
            batch_throughput = opts["micro_batch_size"] * opts["batches_per_step"] / avg_time
            logger.print_to_file_and_screen("epochs:{}, index:{}, loss: {:.4f}, accuracy: {:.4f}, time over batch: {:.4f}, sample/sec: {:.1f}, learning rate: {}".format(epoch, i, loss, accuracy, avg_time, batch_throughput, lr_value), opts)
            i += 1
    train_graph.saver.save(train_graph.session, save_path=path)
    throughput = opts["micro_batch_size"] * iterations / total_time
    tf_log.info(f"Total time:{total_time},sample/sec (averaged over steps):{throughput}")
    train_graph.session.close()


def add_model_arguments(parser):
    parser.add_argument("--max-seq-len", type=int, default=100, help="sequence maximum length")
    parser.add_argument("--hidden-size", type=int, default=36, help="hidden size")
    parser.add_argument("--attention-size", type=int, default=36, help="attention size")
    return parser


def add_dataset_arguments(parser):
    group = parser.add_argument_group('Dataset')
    group.add_argument('--use-synthetic-data', default=False, action='store_true', help='Use synthetic data')
    group.add_argument('--epochs', type=int, default=2, help='number of epochs')
    group.add_argument('--batches-per-step', type=int, default=1600, help='Number of batches to perform on the device before returning to the host')
    return parser


def add_training_arguments(parser):
    group = parser.add_argument_group('Training')
    group.add_argument('--seed', type=int, help="set random seed")
    group.add_argument('--micro-batch-size', type=int, default=32, help="set micro-batch-size for training graph")
    group.add_argument('--replicas', type=int, default=1, help="Replicate graph over N workers to increase batch to micro-batch-size*N")
    group.add_argument('--learning-rate', type=float, default=0.1, help="learning rate")
    group.add_argument('--lr-file', type=str, default='./learning_rate_schedule_file.txt', help='The learning rate file.')
    group.add_argument('--lr-type', type=str, default='fixed', choices=['fixed', 'file'], help="Choose the type of learning rate")
    group.add_argument('--model-path', type=str, default='./dnn_save_path/ckpt_noshuffDIN3', help='Place to store and restore model')
    group.add_argument('--data-type', type=str, default='float32', choices=['float32'], help='Choose the data type.')
    group.add_argument('--large-embedding', default=False, action='store_true', help="set small or large embedding size")
    group.add_argument('--use-ipu-model', default=False, action='store_true', help="use IPU model or not.")
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "CTR Model Training in TensorFlow")
    parser = add_model_arguments(parser)
    parser = add_dataset_arguments(parser)
    parser = add_training_arguments(parser)
    parser = logger.add_arguments(parser)
    args, unknown = parser.parse_known_args()
    args = vars(args)

    seed = args['seed']
    if seed is not None:
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        utils.reset_ipu_seed(seed)
    logger.print_setting(args)
    setup_logger(logging.INFO, tf_log)

    train_process(args)
