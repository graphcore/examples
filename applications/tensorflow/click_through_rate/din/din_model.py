
# Copyright (c) 2020 Graphcore Ltd.
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

"""
CTR Model -- DIN
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ipu import scopes as ipu_scope
from tensorflow.python.ops import init_ops
import logging
from common.Dice import dice

from common.utils import prelu, din_attention

tf_log = logging.getLogger('DIN')


class DIN(object):

    def __init__(self, uid_embedding, mid_embedding, cat_embedding, opts, is_training = True, seed = 3):
        self.model_dtype = tf.float32
        self.HIDDEN_SIZE = opts['hidden_size']
        self.ATTENTION_SIZE = opts['attention_size']
        self.is_training = is_training
        self.use_negsampling = False
        self.opts = opts
        self.EMBEDDING_DIM = 18
        self.n_uid = 543060
        self.n_mid = 367983
        self.n_cat = 1601
        self.uid_embedding = uid_embedding
        self.mid_embedding = mid_embedding
        self.cat_embedding = cat_embedding
        self.glorot = init_ops.glorot_uniform_initializer(seed=seed)

    def look_up(self, embedding, batch_ph, name):
        return embedding.lookup(batch_ph)

    def build_host_embedding_ipu(self):
        self.uid_batch_embedded = self.look_up(self.uid_embedding, self.uid_batch_ph, name = 'uid_embedding_lookup')
        self.mid_batch_embedded = self.look_up(self.mid_embedding, self.mid_batch_ph, name = 'mid_embedding_lookup')
        self.mid_his_batch_embedded = self.look_up(self.mid_embedding, self.mid_his_batch_ph, name = 'mid_his_embedding_lookup')

        self.cat_batch_embedded = self.look_up(self.cat_embedding, self.cat_batch_ph, name = 'cat_embedding_lookup')
        self.cat_his_batch_embedded = self.look_up(self.cat_embedding, self.cat_his_batch_ph, name = 'cat_his_embedding_lookup')

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)


    def build_fcn_net(self, inp, use_dice = False):
        def dtype_getter(getter, name, dtype=None, *args, **kwargs):
            var = getter(name, dtype=self.model_dtype, *args, **kwargs)
            return var

        with tf.compat.v1.variable_scope("fcn", custom_getter=dtype_getter, dtype=self.model_dtype):
            dnn1 = keras.layers.Dense(200, activation=None, kernel_initializer=init_ops.glorot_uniform_initializer(seed=3), name='f1')
            dnn1 = dnn1(inp)
            if use_dice:
                dnn1 = dice(dnn1, name='dice_1', data_type=self.model_dtype)
            else:
                dnn1 = prelu(dnn1, 'prelu1')

            dnn2 = keras.layers.Dense(80, activation=None, kernel_initializer=init_ops.glorot_uniform_initializer(seed=3), name='f2')
            dnn2 = dnn2(dnn1)
            if use_dice:
                dnn2 = dice(dnn2, name='dice_2', data_type=self.model_dtype)
            else:
                dnn2 = prelu(dnn2, 'prelu2')
            dnn3 = keras.layers.Dense(2, activation=None, kernel_initializer=init_ops.glorot_uniform_initializer(seed=3), name='f3')
            dnn3 = dnn3(dnn2)
            y_hat = tf.nn.softmax(dnn3) + 0.00000001
            return y_hat


    def _build_graph_embed(self, uids, mids, cats, mid_his, cat_his, mid_mask, sl, lr, target):
        self.uid_batch_ph = uids
        self.mid_batch_ph = mids
        self.cat_batch_ph = cats
        self.mid_his_batch_ph = mid_his
        self.cat_his_batch_ph = cat_his
        self.mask = mid_mask
        self.seq_len_ph = sl

        self.build_host_embedding_ipu()
        mask = tf.expand_dims(self.mask, axis=2)
        mask = tf.tile(mask, (1, 1, self.EMBEDDING_DIM*2))
        paddings = tf.zeros_like(self.item_his_eb, dtype=self.model_dtype)
        mask = tf.equal(mask, tf.ones_like(mask))
        item_his_eb_masked = tf.where(mask, self.item_his_eb, paddings)
        self.item_his_eb_sum = tf.reduce_sum(item_his_eb_masked, 1)

        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.item_eb, item_his_eb_masked,  self.ATTENTION_SIZE, self.mask, kernel_initializer=self.glorot)
            att_fea = tf.reduce_sum(attention_output, 1)
        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, att_fea], -1)
        # Fully connected layer
        y_hat = self.build_fcn_net(inp, use_dice=True)

        loss = - tf.reduce_mean(tf.math.log(y_hat) * target)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(y_hat), target), self.model_dtype))

        if self.is_training:
            tf_log.info("optimizer is SGD")
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=lr)

            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars)
        else:
            train_op = None

        return y_hat, loss, accuracy, train_op

    def build_graph(self, uids, mids, cats, mid_his, cat_his, mid_mask, sl, lr, target):
        return self._build_graph_embed(uids, mids, cats, mid_his, cat_his, mid_mask, sl, lr, target)

    def __call__(self, uids, mids, cats, mid_his, cat_his, mid_mask, sl, lr, target):
        return self.build_graph(uids, mids, cats, mid_his, cat_his, mid_mask, sl, lr, target)
