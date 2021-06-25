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
import tensorflow as tf
from collections import namedtuple
from tensorflow.python.ipu import utils
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import embedding_ops
from tensorflow.python.ipu.config import IPUConfig
import set_path
from common.utils import calc_auc, setup_logger
from common.embedding import get_dataset_embed_from_tensors, id_embedding, get_synthetic_dataset
import common.log as logger
from dien.dien_model import DIEN

TRAIN_DATA_SIZE = 1086120
tf_log = logging.getLogger('DIEN')


GraphOps = namedtuple(
    'graphOps', ['session',
                 'init',
                 'ops',
                 'placeholders',
                 'iterator',
                 'saver',
                 'feed_dict_values'])


def get_tf_datatype(opts):
    dtypes = opts["precision"].split('.')
    master_dtype = tf.float16 if dtypes[1] == '16' else tf.float32
    return master_dtype


def graph_builder(opts, uid_embedding, mid_embedding, cat_embedding, lr,  uids, mids, cats, mid_his, cat_his, mid_mask, target, seqlen, noclk_mids, noclk_cats, use_negsampling=True):
    master_dtype = get_tf_datatype(opts)
    return DIEN(opts, uid_embedding, mid_embedding, cat_embedding, master_dtype, is_training=True, use_negsampling = True, optimizer=opts['optimizer'])(uids, mids, cats, mid_his, cat_his, mid_mask, seqlen, noclk_mids, noclk_cats, lr, target)


def generic_graph(opts):
    data_type = get_tf_datatype(opts)
    graph = tf.Graph()
    with graph.as_default():
        placeholders = {}
        placeholders["learning_rate"] = tf.placeholder(data_type, shape=[])
        uid_embedding, mid_embedding, cat_embedding = id_embedding(opts, True, opts['seed'])
        if opts['use_synthetic_data']:
            dataset = get_synthetic_dataset(opts, return_neg=True)
            feed_dict_values = {}
        else:
            dataset, feed_dict_values = get_dataset_embed_from_tensors(opts, data_type)
        infeed = ipu_infeed_queue.IPUInfeedQueue(dataset, feed_name = 'DIEN_dataset_infeed', replication_factor = (opts['replicas']))

        with ipu_scope('/device:IPU:0'):
            def comp_fn():
                def body(total_loss, total_aux_loss, total_accuracy, uids, mids, cats, mid_his, cat_his, mid_mask, target, seqlen,  noclk_mids, noclk_cats):
                    prob, loss, aux_loss, accuracy, grad_op = graph_builder(opts, uid_embedding, mid_embedding, cat_embedding, placeholders['learning_rate'], uids, mids, cats, mid_his, cat_his, mid_mask, target, seqlen, noclk_mids, noclk_cats, use_negsampling=True)
                    with tf.control_dependencies([grad_op]):
                        return total_loss + loss, total_aux_loss + aux_loss, total_accuracy + accuracy
                return loops.repeat(opts['batches_per_step'], body, [tf.constant(0, data_type)]*3, infeed)
            outputs_train = ipu_compiler.compile(comp_fn, [])
            avg_loss, avg_aux_loss, avg_accuracy = [x / opts['batches_per_step'] for x in outputs_train]

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
    utils.reset_ipu_seed(opts['seed'])

    graph_outputs = [avg_loss, avg_aux_loss, avg_accuracy]
    sess = tf.Session(graph=graph)

    return GraphOps(sess,
                    init,
                    graph_outputs,
                    placeholders,
                    infeed,
                    saver,
                    feed_dict_values), uid_embedding, mid_embedding, cat_embedding


def generic_init_graph(opts):
    graph_ops, uid_embedding, mid_embedding, cat_embedding = generic_graph(opts)
    graph_ops.session.run(graph_ops.init)
    graph_ops.session.run(graph_ops.iterator.initializer, feed_dict=graph_ops.feed_dict_values)
    return graph_ops, uid_embedding, mid_embedding, cat_embedding


def train_process(opts, restore=False):
    path = opts['model_path']
    graph, uid_embedding, mid_embedding, cat_embedding = generic_init_graph(opts)
    epochs = opts['epochs']
    iterations_per_epoch = TRAIN_DATA_SIZE / (opts['batch_size'] * opts['replicas'])
    steps = epochs * iterations_per_epoch // opts["batches_per_step"]

    total_time = 0
    i = 0
    lr = opts['learning_rate']
    tf_log.debug(f"steps: {steps}")
    with uid_embedding.register(graph.session), mid_embedding.register(graph.session), cat_embedding.register(graph.session):
        while i < steps:
            start = time.time()
            loss, aux_loss, accuracy = graph.session.run(graph.ops, feed_dict = {graph.placeholders["learning_rate"]: lr})
            time_one_iteration = time.time() - start
            if i == 1:
                total_time += time_one_iteration * 2
            if i > 1:
                total_time += time_one_iteration
            batch_throughput = opts["batch_size"] * opts["batches_per_step"] / time_one_iteration
            logger.print_to_file_and_screen("index:{}, loss: {:.4f}, aux_loss: {:.4f}, accuracy: {:.4f}, time over batch: {:.4f}, sample/sec: {:.1f}, learning rate: {}".format(i, loss, aux_loss, accuracy, time_one_iteration, batch_throughput, lr), opts)
            i += 1

    if steps > 1:
        throughput = steps * opts["batches_per_step"] * opts['batch_size'] / total_time
        tf_log.debug(f"TTT:{total_time}, Samples/second (averaged over steps):{throughput}")
    graph.saver.save(graph.session, save_path=path)
    graph.session.close()


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
    group.add_argument('--batch-size', type=int, default=32, help = "set batch-size for training graph")
    group.add_argument('--replicas', type=int, default=1, help = "Replicate graph over N workers to increase batch to batch-size*N")
    group.add_argument('--optimizer', type=str, default="SGD", choices=['SGD', 'Adam'], help="optimizer")
    group.add_argument('--learning-rate', type=float, default=0.6, help = "learning rate")
    group.add_argument('--large-embedding', default=False, action='store_true', help="set small or large embedding size")
    group.add_argument('--model-path', type=str, default='./dnn_save_path/ckpt_noshuffDIEN3', help='Place to store and restore model')
    group.add_argument('--use-ipu-model', default=False, action='store_true', help="use IPU model or not.")
    group.add_argument('--use-ipu-emb', default=False, action='store_true', help = "Use host embeddig or put embedding on ipu.")
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "CTR Model Training in Tensorflow", add_help = False)
    parser = add_model_arguments(parser)
    parser = add_dataset_arguments(parser)
    parser = add_training_arguments(parser)
    parser = logger.add_arguments(parser)
    args, _ = parser.parse_known_args()
    args = vars(args)
    logger.print_setting(args, is_dien=False, is_training=True)
    setup_logger(logging.DEBUG, tf_log, name='dien_log.txt')

    seed = args['seed']
    if seed is not None:
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        utils.reset_ipu_seed(seed)

    train_process(args)
