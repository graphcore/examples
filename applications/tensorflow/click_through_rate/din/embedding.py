# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

# Copyright 1999-present Alibaba Group Holding Ltd.
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
#
# This file has been modified by Graphcore Ltd.
# It has been modified to run the application on IPU hardware.

import numpy as np
import tensorflow as tf
import logging
from din.data_generation import data_generator
from tensorflow.python.ipu import embedding_ops

EMBEDDING_DIM = 18
TRAIN_DATA_SIZE = 1086120
VALIDATION_DATA_SIZE = 121216
STORE_LAST_MODELS_NUM = 10
use_host_eb = True
best_auc = 0.0
lowlen_global = None
highlen_global = None

tf_log = logging.getLogger('DIN')


def get_synthetic_dataset(opts):
    N_CAT = 1601
    N_UID = 543060
    N_MID = 30000000 if opts['large_embedding'] else 367983

    datatype = 'float32'
    mid_his = tf.random.uniform([opts['max_seq_len'], ],
                                minval = 0,
                                maxval= N_MID,
                                dtype = tf.int32,
                                name='mid_his')

    cat_his = tf.random.uniform([opts['max_seq_len'], ],
                                minval = 0,
                                maxval= N_CAT,
                                dtype = tf.int32,
                                name='cat_his')

    uids = tf.random.uniform([],
                             minval = 0,
                             maxval= N_UID,
                             dtype = tf.int32,
                             name='uid')

    mids = tf.random.uniform([],
                             minval = 0,
                             maxval= N_MID,
                             dtype = tf.int32,
                             name='mid')

    cats = tf.random.uniform([],
                             minval = 0,
                             maxval= N_CAT,
                             dtype = tf.int32,
                             name='cat')

    mid_mask = tf.random.uniform([opts['max_seq_len'], ],
                                 minval = 0,
                                 maxval= 1,
                                 dtype = datatype,
                                 name='mask')

    target = tf.random.uniform([2, ],
                               minval = 0,
                               maxval= 1,
                               dtype = datatype,
                               name='target_ph')

    sl = tf.constant(opts['max_seq_len'])

    dataset = tf.data.Dataset.from_tensors((uids,
                                            mids,
                                            cats,
                                            mid_his,
                                            cat_his,
                                            mid_mask,
                                            target,
                                            sl
                                            ))

    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.batch(opts['batch_size'], drop_remainder=True)
    dataset = dataset.prefetch(1024)

    return dataset


def get_dataset_embed(opts, is_training=True):
    data_type = 'float32'
    dataset = tf.data.Dataset.from_generator(lambda: data_generator(opts, is_training),
                                             (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, data_type, data_type, tf.int32),
                                             (tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([opts["max_seq_len"]]), tf.TensorShape([opts["max_seq_len"]]),
                                              tf.TensorShape([opts["max_seq_len"]]), tf.TensorShape([2]), tf.TensorShape([])))

    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.batch(opts['batch_size'], drop_remainder=True)
    dataset = dataset.prefetch(1024)
    tf_log.info(f"batch_size={opts['batch_size']}")
    return dataset


def build_embeddings(opts, name, shape, is_training, seed_b):
    data_type = 'float32'
    if is_training:
        optimizer_spec = embedding_ops.HostEmbeddingOptimizerSpec(opts["learning_rate"])
    else:
        optimizer_spec = None
    return embedding_ops.create_host_embedding(name, shape=shape, dtype=getattr(np, data_type), optimizer_spec=optimizer_spec, initializer=tf.keras.initializers.glorot_uniform(seed=seed_b))


def id_embedding(opts, is_training, seed):
    uid_embedding = build_embeddings(opts, "uid_embedding", [543060, EMBEDDING_DIM], is_training, seed)
    mid_embedding = build_embeddings(opts, "mid_embedding", [367983, EMBEDDING_DIM], is_training, seed)
    cat_embedding = build_embeddings(opts, "cat_embedding", [1601, EMBEDDING_DIM], is_training, seed)
    return uid_embedding, mid_embedding, cat_embedding
