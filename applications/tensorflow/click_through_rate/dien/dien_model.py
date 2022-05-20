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

"""
CTR Model -- DIEN
"""

import tensorflow.compat.v1 as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops import init_ops
from tensorflow.python.ipu.ops.rnn_ops import PopnnAUGRU
from tensorflow.python.ipu.ops.rnn_ops import PopnnDynamicGRU
from tensorflow.python.ipu.ops.embedding_ops import embedding_lookup as ipu_embedding_lookup
from dien.rnn import dynamic_rnn
from common.utils import din_fcn_attention, prelu, VecAttGRUCell
from common.Dice import dice


class DIEN(object):

    def __init__(self,
                 opts,
                 uid_embedding,
                 mid_embedding,
                 cat_embedding,
                 data_type,
                 is_training = False,
                 use_negsampling = False,
                 optimizer = None):
        self.model_dtype = data_type
        self.opts = opts
        self.hidden_size = opts['hidden_size']
        self.attention_size = opts['attention_size']
        self.gru_type = opts['gru_type']
        self.augru_type = opts['augru_type']
        self.maxlen = opts['max_seq_len']
        self.optimizer_type = optimizer
        self.micro_batch_size = opts['micro_batch_size']
        self.use_ipu_emb = opts['use_ipu_emb']
        self.embedding_dim = 18
        self.use_negsampling = use_negsampling
        self.is_training = is_training
        self.uid_embedding = uid_embedding
        self.mid_embedding = mid_embedding
        self.cat_embedding = cat_embedding
        self.n_uid = 543060
        self.n_mid = 367983
        self.n_cat = 1601
        self.glorot = init_ops.glorot_uniform_initializer(seed=opts['seed'], dtype=self.model_dtype)


    def build_embedding(self):
        if not self.use_ipu_emb:
            self.uid_batch_embedded = self.uid_embedding.lookup(self.uid_batch_ph)
            self.mid_batch_embedded = self.mid_embedding.lookup(self.mid_batch_ph)
            self.mid_his_batch_embedded = self.mid_embedding.lookup(self.mid_his_batch_ph)
            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = self.mid_embedding.lookup(self.noclk_mid_batch_ph)
                self.noclk_cat_his_batch_embedded = self.cat_embedding.lookup(self.noclk_cat_batch_ph)
            self.cat_batch_embedded = self.cat_embedding.lookup(self.cat_batch_ph)
            self.cat_his_batch_embedded = self.cat_embedding.lookup(self.cat_his_batch_ph)
        else:
            with tf.variable_scope('embedding_layer', use_resource=True, reuse=tf.AUTO_REUSE):
                self.uid_embeddings_var = tf.get_variable("uid_embedding", shape=[self.n_uid, self.embedding_dim], dtype=tf.float32)
                self.uid_batch_embedded = ipu_embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph, name='uid_embedding_lookup')
                self.mid_embeddings_var = tf.get_variable("mid_embedding", shape=[self.n_mid, self.embedding_dim], dtype=tf.float32)
                self.mid_batch_embedded = ipu_embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph, name='mid_embedding_lookup')
                self.mid_his_batch_embedded = ipu_embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph, name='mid_his_embedding_lookup')
                self.cat_embeddings_var = tf.get_variable("cat_embedding", shape=[self.n_cat, self.embedding_dim], dtype=tf.float32)
                self.cat_batch_embedded = ipu_embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph, name='cat_embedding_lookup')
                self.cat_his_batch_embedded = ipu_embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph, name='cat_his_embedding_lookup')
                if self.use_negsampling:
                    self.noclk_mid_his_batch_embedded = ipu_embedding_lookup(self.mid_embeddings_var, self.noclk_mid_batch_ph, name='noclk_mid_his_embedding_lookup')
                    self.noclk_cat_his_batch_embedded = ipu_embedding_lookup(self.cat_embeddings_var, self.noclk_cat_batch_ph, name='noclk_cat_his_embedding_lookup')

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)

        if self.use_negsampling:
            self.noclk_item_his_eb = tf.concat(
                [self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cat_his_batch_embedded[:, :, 0, :]], -1)  # 0 means only using the first negative item ID. 3 item IDs are inputed in the line 24.
            self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb,
                                                [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1], 36])  # cat embedding 18 concate item embedding 18.
            self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cat_his_batch_embedded], -1)
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)


    def build_fcn_net(self, inp, use_dice = False):
        def dtype_getter(getter, name, dtype=None, *args, **kwargs):
            return getter(name, dtype=self.model_dtype, *args, **kwargs)

        with tf.variable_scope("fcn", custom_getter=dtype_getter, dtype=self.model_dtype):
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
            dnn1 = tf.layers.dense(bn1, 200, kernel_initializer=self.glorot, activation=None, name='f1')
            if use_dice:
                dnn1 = dice(dnn1, name='dice_1', data_type=self.model_dtype)
            else:
                dnn1 = prelu(dnn1, 'prelu1')
            dnn2 = tf.layers.dense(dnn1, 80, kernel_initializer=self.glorot, activation=None, name='f2')
            if use_dice:
                dnn2 = dice(dnn2, name='dice_2', data_type=self.model_dtype)
            else:
                dnn2 = prelu(dnn2, 'prelu2')
            dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
            y_hat = tf.nn.softmax(dnn3) + 0.00000001
            return y_hat


    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag = None):
        def dtype_getter(getter, name, dtype=None, *args, **kwargs):
            var = getter(name, dtype=self.model_dtype, *args, **kwargs)
            return var

        with tf.variable_scope("aux_loss", custom_getter=dtype_getter, dtype=self.model_dtype):
            mask = tf.cast(mask, self.model_dtype)
            click_input_ = tf.concat([h_states, click_seq], -1)
            noclick_input_ = tf.concat([h_states, noclick_seq], -1)
            click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
            noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]
            click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
            noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
            loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
            loss_ = self.micro_batch_size * self.maxlen / tf.reduce_sum(mask) * loss_
            return loss_


    def auxiliary_net(self, in_, stag='auxiliary_net'):
        def dtype_getter(getter, name, dtype=None, *args, **kwargs):
            var = getter(name, dtype=self.model_dtype, *args, **kwargs)
            return var

        with tf.variable_scope("aux_net", custom_getter=dtype_getter, dtype=self.model_dtype):
            bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
            dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
            dnn1 = tf.nn.sigmoid(dnn1)
            dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
            dnn2 = tf.nn.sigmoid(dnn2)
            dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
            y_hat = tf.nn.softmax(dnn3) + 0.00000001
            return y_hat


    def _build_graph_embed(self, uids, mids, cats, mid_his, cat_his, mid_mask, sl, noclk_mids, noclk_cats, lr, target):
        self.uid_batch_ph = uids
        self.mid_batch_ph = mids
        self.cat_batch_ph = cats
        self.mid_his_batch_ph = mid_his
        self.cat_his_batch_ph = cat_his
        self.noclk_mid_batch_ph = noclk_mids
        self.noclk_cat_batch_ph = noclk_cats
        self.mask = mid_mask
        self.seq_len_ph = sl

        def dtype_getter(getter, name, dtype=None, *args, **kwargs):
            var = getter(name, dtype=self.model_dtype, *args, **kwargs)
            return var

        self.build_embedding()
        mask = tf.expand_dims(self.mask, axis=2)
        mask = tf.tile(mask, (1, 1, self.embedding_dim*2))
        paddings = tf.zeros_like(self.item_his_eb, dtype=self.model_dtype)
        mask = tf.equal(mask, tf.ones_like(mask))
        self.item_his_eb = tf.where(mask, self.item_his_eb, paddings)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)

        if self.use_negsampling:
            self.noclk_item_his_eb = tf.where(mask, self.noclk_item_his_eb, paddings)

        with tf.variable_scope("dien", custom_getter=dtype_getter, dtype=self.model_dtype):
            # RNN layer
            with tf.name_scope('rnn_1'):

                if self.gru_type == "TfnnGRU":
                    rnn_outputs, _ = dynamic_rnn(GRUCell(self.hidden_size), inputs=self.item_his_eb,
                                                 sequence_length=self.seq_len_ph, dtype=self.model_dtype,
                                                 scope="gru1")
                elif self.gru_type == "PopnnGRU":
                    gru = PopnnDynamicGRU(self.hidden_size)
                    rnn_outputs, _ = gru(self.item_his_eb, self.seq_len_ph, time_major=False)
                    rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
            if self.use_negsampling:
                aux_loss = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                               self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")

            # Attention layer
            with tf.name_scope('Attention_layer_1'):
                att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, self.attention_size, self.mask,
                                                        softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)

            with tf.name_scope('rnn_2'):
                if self.augru_type == "TfAUGRU":
                    rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(self.hidden_size), inputs=rnn_outputs,
                                                             att_scores = tf.expand_dims(alphas, -1),
                                                             sequence_length=self.seq_len_ph, dtype=self.model_dtype,
                                                             scope="gru2")
                elif self.augru_type == "PopnnAUGRU":
                    augru = PopnnAUGRU(self.hidden_size)
                    rnn_outputs2, final_state2 = augru(rnn_outputs, self.seq_len_ph, alphas, time_major=False)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2], 1)
        y_hat = self.build_fcn_net(inp, use_dice=True)

        # loss
        loss = - tf.reduce_mean(tf.log(y_hat) * target)
        if self.use_negsampling:
            loss = loss + aux_loss
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(y_hat), target), self.model_dtype))
        if self.is_training:
            if self.optimizer_type == 'SGD':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
            elif self.optimizer_type == "Adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            else:
                raise Exception("No optimizer is specified.")
            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars)
            return y_hat, loss, aux_loss, accuracy, train_op
        else:
            return y_hat, accuracy


    def __call__(self, uids, mids, cats, mid_his, cat_his, mid_mask, sl, noclk_mids, noclk_cats, lr, target):
        return self._build_graph_embed(uids, mids, cats, mid_his, cat_his, mid_mask, sl, noclk_mids, noclk_cats, lr, target)
