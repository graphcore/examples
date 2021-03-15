# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#
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
# This file includes modified code from the file macro_benchmark/DIEN/script/train.py in the AI-matrix repository.

import numpy as np
import logging
from common.data_iterator import DataIterator

EMBEDDING_DIM = 18
TRAIN_DATA_SIZE = 1086120
VALIDATION_DATA_SIZE = 121216
STORE_LAST_MODELS_NUM = 10

tf_log = logging.getLogger('common')


def prepare_data(opts, input, target, maxlen = None, return_neg = False):
    # x: a list of sentences
    lengths_x = [len(s[4]) for s in input]
    seqs_mid = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]
    noclk_seqs_mid = [inp[5] for inp in input]
    noclk_seqs_cat = [inp[6] for inp in input]
    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)
    maxlen_x = np.max(lengths_x)
    neg_samples = len(noclk_seqs_mid[0][0])

    mid_his = np.zeros((n_samples, maxlen)).astype('int64')
    cat_his = np.zeros((n_samples, maxlen)).astype('int64')
    noclk_mid_his = np.zeros((n_samples, maxlen, neg_samples)).astype('int64')
    noclk_cat_his = np.zeros((n_samples, maxlen, neg_samples)).astype('int64')
    data_type = 'float32'
    mid_mask = np.zeros((n_samples, maxlen)).astype(data_type)
    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

    uids = np.array([inp[0] for inp in input])
    mids = np.array([inp[1] for inp in input])
    cats = np.array([inp[2] for inp in input])

    if return_neg:
        return uids, mids, cats, mid_his, cat_his, mid_mask, np.array(target), np.array(lengths_x), noclk_mid_his, noclk_cat_his
    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, np.array(target), np.array(lengths_x)


def data_generator(opts, is_training, return_neg=False):
    if is_training:
        file = "./common/local_train_splitByUser"
        data_size = TRAIN_DATA_SIZE
    else:
        file = "./common/local_test_splitByUser"
        data_size = VALIDATION_DATA_SIZE
    uid_voc = "./common/uid_voc.pkl"
    mid_voc = "./common/mid_voc.pkl"
    cat_voc = "./common/cat_voc.pkl"
    batch_size = opts['batch_size']

    data_itr = DataIterator(file, uid_voc, mid_voc, cat_voc, batch_size, opts["max_seq_len"], shuffle_each_epoch=False)
    tf_log.info(f"data n: {data_itr.get_n()}")
    i = 0
    for src, tgt in data_itr:
        i += batch_size
        uids, mids, cats, mid_his, cat_his, mid_mask, target, seqlen = prepare_data(opts, src, tgt, opts["max_seq_len"], return_neg=return_neg)
        if i >= data_size:
            raise StopIteration
        if len(uids) < opts['batch_size']:
            raise StopIteration
        for j in range(batch_size):
            yield np.squeeze(np.array(uids[j])), np.squeeze(np.array(mids[j])), np.squeeze(np.array(cats[j])), np.squeeze(mid_his[j]), np.squeeze(cat_his[j]), np.squeeze(mid_mask[j]), np.squeeze(np.array(target[j])), np.squeeze(np.array(seqlen[j]))


def parse_data(opts):
    data = {'uids': [], 'mids': [], 'cats': [], 'mid_his': [], 'cat_his': [], 'mid_mask': [], 'target': [], 'seqlen': [], 'noclk_mids': [], 'noclk_cats': []}

    file = "./common/local_train_splitByUser"
    uid_voc = "./common/uid_voc.pkl"
    mid_voc = "./common/mid_voc.pkl"
    cat_voc = "./common/cat_voc.pkl"
    batch_size = opts['batch_size']

    data_itr = DataIterator(file, uid_voc, mid_voc, cat_voc, batch_size, opts["max_seq_len"], shuffle_each_epoch=False)
    tf_log.debug("Start to prepare data.")
    i = 0
    for src, tgt in data_itr:
        i += batch_size
        items = prepare_data(opts, src, tgt, opts["max_seq_len"], return_neg=True)
        for j in range(len(items[0])):
            for i, key in enumerate(data.keys()):
                data[key].append(np.squeeze(np.array(items[i][j])).tolist())
        if i >= TRAIN_DATA_SIZE:
            break
    tf_log.debug("End to prepare data.")
    return [v for v in data.values()]
