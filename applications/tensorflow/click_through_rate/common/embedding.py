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

import os
import numpy as np
import tensorflow.compat.v1 as tf
import logging
from tensorflow.python.ipu import embedding_ops
from common.data_generation import data_generator, parse_data

EMBEDDING_DIM = 18
tf_log = logging.getLogger('common')


def get_dataset_embed_from_tensors(opts, data_type):
    parsed_data_path = 'train_parsed_data_bs{}_maxlen{}.npz'.format(opts['batch_size'], opts['max_seq_len'])
    if os.path.exists(parsed_data_path):
        tf_log.debug('Use parsed data.')
        data = np.load(parsed_data_path)
        uids = data['uids']
        mids = data['mids']
        cats = data['cats']
        mid_his = data['mid_his']
        cat_his = data['cat_his']
        mid_mask = data['mid_mask']
        target = data['target']
        seqlen = data['seqlen']
        noclk_mids = data['noclk_mids']
        noclk_cats = data['noclk_cats']
    else:
        uids, mids, cats, mid_his, cat_his, mid_mask, target, seqlen, noclk_mids, noclk_cats = parse_data(opts)
        np.savez(parsed_data_path, uids=uids, mids=mids, cats=cats, mid_his=mid_his, cat_his=cat_his, mid_mask=mid_mask, target=target, seqlen=seqlen, noclk_mids=noclk_mids, noclk_cats=noclk_cats)

    size = len(uids)
    uids_placeholder = tf.placeholder(dtype=tf.int32, shape=[size], name='uids_placeholder')
    mids_placeholder = tf.placeholder(tf.int32, [size])
    cats_placeholder = tf.placeholder(tf.int32, [size])
    mid_his_placeholder = tf.placeholder(tf.int32, [size, opts["max_seq_len"]])
    cat_his_placeholder = tf.placeholder(tf.int32, [size, opts["max_seq_len"]])
    mid_mask_placeholder = tf.placeholder(data_type, [size, opts["max_seq_len"]])
    target_placeholder = tf.placeholder(data_type, [size, 2])
    seqlen_placeholder = tf.placeholder(tf.int32, [size])
    noclk_mids_placeholder = tf.placeholder(tf.int32, [size, opts["max_seq_len"], 5])
    noclk_cats_placeholder = tf.placeholder(tf.int32, [size, opts["max_seq_len"], 5])
    placeholders = [uids_placeholder, mids_placeholder, cats_placeholder, mid_his_placeholder, cat_his_placeholder, mid_mask_placeholder, target_placeholder, seqlen_placeholder, noclk_mids_placeholder, noclk_cats_placeholder]
    values = [uids, mids, cats, mid_his, cat_his, mid_mask, target, seqlen, noclk_mids, noclk_cats]
    dataset = tf.data.Dataset.from_tensor_slices(tuple(placeholders))
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.batch(opts['batch_size'], drop_remainder=True)
    dataset = dataset.prefetch(1024)

    feed_dict_values = {}
    for placeholder, value in zip(placeholders, values):
        feed_dict_values[placeholder] = value

    return dataset, feed_dict_values


def get_synthetic_dataset(opts, return_neg=False):
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

    if return_neg:
        noclk_mids = tf.random.uniform([opts['max_seq_len'], 5],
                                       minval = 0,
                                       maxval= N_MID,
                                       dtype = tf.int32,
                                       name='noclk_mid_batch_ph')

        noclk_cats = tf.random.uniform([opts['max_seq_len'], 5],
                                       minval = 0,
                                       maxval= N_CAT,
                                       dtype = tf.int32,
                                       name='noclk_cat_batch_ph')

        dataset = tf.data.Dataset.from_tensors((uids,
                                                mids,
                                                cats,
                                                mid_his,
                                                cat_his,
                                                mid_mask,
                                                target,
                                                sl,
                                                noclk_mids,
                                                noclk_cats
                                                ))
    else:
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


def get_dataset_embed(opts, is_training=True, return_neg=False, data_type = 'float32'):
    dataset = tf.data.Dataset.from_generator(lambda: data_generator(opts, is_training, return_neg=return_neg),
                                             (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, data_type, data_type, tf.int32),
                                             (tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([opts["max_seq_len"]]), tf.TensorShape([opts["max_seq_len"]]),
                                              tf.TensorShape([opts["max_seq_len"]]), tf.TensorShape([2]), tf.TensorShape([])))

    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.batch(opts['batch_size'], drop_remainder=True)
    dataset = dataset.prefetch(1024)
    tf_log.info(f"batch_size={opts['batch_size']}")
    return dataset


def build_embeddings(opts, name, shape, is_training, seed_b, data_type = 'float32'):
    if is_training:
        optimizer_spec = embedding_ops.HostEmbeddingOptimizerSpec(opts["learning_rate"])
    else:
        optimizer_spec = None
    return embedding_ops.create_host_embedding(name, shape=shape, dtype=getattr(np, data_type), optimizer_spec=optimizer_spec, initializer=tf.keras.initializers.glorot_uniform(seed=seed_b))


def id_embedding(opts, is_training, seed, data_type = 'float32'):
    uid_embedding = build_embeddings(opts, "uid_embedding", [543060, EMBEDDING_DIM], is_training, seed, data_type)
    mid_embedding = build_embeddings(opts, "mid_embedding", [367983, EMBEDDING_DIM], is_training, seed, data_type)
    cat_embedding = build_embeddings(opts, "cat_embedding", [1601, EMBEDDING_DIM], is_training, seed, data_type)
    return uid_embedding, mid_embedding, cat_embedding
