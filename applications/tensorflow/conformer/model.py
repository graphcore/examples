# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
import yaml
import math
import editdistance
import numpy as np
import tensorflow as tf

from itertools import groupby
from functools import partial

from tensorflow.python.framework import graph_util
from tensorflow.python import ipu
from tensorflow.python.ipu import embedding_ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu.ops import pipelining_ops

from ipu_optimizer import AdamLossScalingOptimizer
from data_loader import Dataloader
from util import get_config


class AMConfig(object):
    """Configuration for `Conformer Acoustic Model`"""

    def __init__(self):
        self.config = {
            'is_training': False,
            'dtype': 'FLOAT16',
            'batch_size': 1,
            'maxlen_in': 512,
            'maxlen_tgt': 48,
            'fbank_size': 83,
            'adim': 256,
            'elayers': 12,
            'eunits': 2048,
            'dropout_rate': 0.0,
            'aheads': 4,
            'kernel_size': 15,
            'dlayers': 1,
            'dunits': 512,
            'lr': 0.0,
            'warmup_steps': 20000,
            'attn_dropout_rate': 0.0,
            'epochs': 10,
            'vocab_size': 4233,
            'optimizer': 'sgd',
            'replica': 1,
            'mtlalpha': 0.3,
            'lsm_weight': 0.1,
            'data_path': None,
            'dict_path': None,
            'gradient_accumulation_count': 12,
            'loss_scale': 512,
            'use_synthetic_data': False,
            'wandb_name': None,
            'use_ipu_dropout': True,
            'initializer_range': 0.083
        }

    @classmethod
    def from_yaml(cls, yaml_path, **other_args):
        obj = AMConfig()
        if not os.path.exists(yaml_path):
            raise IOError(f"Configuration file {yaml_path}  does not exist. Exiting.")
        yaml_config = yaml.safe_load(open(yaml_path))
        for key, value in other_args.items():
            yaml_config[key] = value
        for key, value in yaml_config.items():
            obj.config[key] = value
        return obj.config


def glu(a, b):
    return tf.math.multiply(a, tf.math.sigmoid(b))


def swish(x):
    return tf.math.multiply(x, tf.math.sigmoid(x))


def create_initializer(maxval=0.02, dtype=tf.float16):
    return tf.compat.v1.random_uniform_initializer(minval=-maxval, maxval=maxval, dtype=dtype)


class ConformerAM(object):
    def __init__(self, config):
        self.config = config
        print(self.config)
        self.infeed_queue = None
        self.outfeed_queue = None
        self.dtype = tf.float16 if self.config['dtype'] == 'FLOAT16' \
            else tf.float32
        self.output_names = []
        self.training = True if self.config['is_training'] else False

        self.computational_stages = []
        self.device_mapping = []

        self.pipeline = []
        self.outputs = ['loss', 'logits', 'labels',
                        'mask', 'kl_loss', 'ctc_loss']
        self.data_loader = None
        self.char_dict = {}

        self.kernel_regularizer = None
        self.bias_regularizer = None
        self.mask_value = -10000


    def save_pb(self, session, output_names):
        graph_def = tf.compat.v1.get_default_graph().as_graph_def()
        graph_def = graph_util.convert_variables_to_constants(
            session, graph_def, output_names)

        with tf.gfile.FastGFile('logs/model.pb', mode='wb') as f:
            f.write(graph_def.SerializeToString())

    def get_lr(self, global_step):
        lr = self.config['lr'] * self.config['adim'] ** (-0.5) * min(global_step ** (-0.5), global_step * self.config['warmup_steps'] ** (-1.5))
        return lr

    def check_loss(self, tgt, loss):
        mask = np.where(tgt == 0, 1, 0).astype(loss.dtype)
        mask = np.expand_dims(mask, axis=-1)
        s = np.sum(mask * loss)

        return s

    def get_kl_acc(self, y_pred, y_true, ignore_id=0):
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        mask = y_true != ignore_id
        numerator = np.sum(y_pred[mask] == y_true[mask])
        denominator = np.sum(mask)

        return float(numerator) / float(denominator)

    def get_ctc_acc(self, y_pred, y_true, blank_id=0):
        y_pred = np.reshape(y_pred, [-1, y_pred.shape[-1]])
        y_true = np.reshape(y_true, [-1, y_true.shape[-1]])

        cers, char_ref_lens = [], []
        for i, y in enumerate(y_pred):
            y_hat_i = [x[0] for x in groupby(y)]
            y_true_i = y_true[i]
            seq_hat, seq_true = [], []
            for idx in y_hat_i:
                idx = int(idx)
                if idx in self.char_dict.keys():
                    seq_hat.append(self.char_dict[int(idx)])

            for idx in y_true_i:
                idx = int(idx)
                if idx in self.char_dict.keys():
                    seq_true.append(self.char_dict[int(idx)])

            hyp_chars = "".join(seq_hat)
            ref_chars = "".join(seq_true)
            if len(ref_chars) > 0:
                cers.append(editdistance.eval(hyp_chars, ref_chars))
                char_ref_lens.append(len(ref_chars))

        cer_ctc = float(sum(cers)) / sum(char_ref_lens) if cers else None

        return cer_ctc

    def _build_dataset(self):
        if not self.config['use_synthetic_data']:
            with open(self.config['dict_path'], 'r') as fp:
                for item in fp.readlines():
                    item = item.strip().split(' ')
                    self.char_dict[int(item[1])] = item[0]

        self.data_loader = Dataloader(self.config['data_path'],
                                      self.config['maxlen_in'],
                                      self.config['maxlen_tgt'],
                                      self.config['vocab_size'],
                                      self.config['fbank_size'],
                                      training=self.training,
                                      dtype=self.config['dtype'],
                                      use_synthetic_data=self.config['use_synthetic_data'])
        self.data_loader.load_data()

        output_types = (self.dtype, tf.int32, tf.int32, tf.int32)
        output_shapes = (
            tf.TensorShape([self.config['maxlen_in'], 83, 1]),
            tf.TensorShape([]),
            tf.TensorShape([self.config['maxlen_tgt']]),
            tf.TensorShape([])
        )
        dataset = tf.data.Dataset.from_generator(self.data_loader, output_types,
                                                 output_shapes=output_shapes)
        dataset = dataset.batch(self.config['batch_size'], drop_remainder=True)

        self.infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
            dataset, prefetch_depth=15)

    def _build_pos_embedding(self, max_len, dmodel, reverse=False):
        '''
        Args:
            max_len: max_len of position embedding
            dmodel: dimension of the position embedding
            reverse: Whether to reverse the input position

        Returns:
            position embedding: (1, max_len, dmodel), untrainable
        '''
        if reverse:
            pos = tf.range(max_len - 1, -1, -1.0, dtype=tf.float32)
        else:
            pos = tf.range(0, max_len, 1.0, dtype=tf.float32)

        index = tf.range(0, dmodel, 2.0, dtype=tf.float32)
        index = 1 / tf.pow(10000.0, (index / dmodel))

        sinusoid = tf.expand_dims(tf.einsum("i,j->ij", pos, index), axis=2)
        pos_emb = tf.concat([tf.sin(sinusoid), tf.cos(sinusoid)], axis=-1)
        pos_emb = tf.reshape(pos_emb, [1, pos_emb.shape[0], -1])

        return tf.cast(pos_emb, self.dtype)

    def _build_encoder_embedding(self, input, input_len, scope_name=None):
        '''
        Args:
            input: 4D-tensor, (batch_size, max_wav_len, feat_dim, 1)
            input_len: 1D-tensor, (batch_size,), valid length for each item

        Returns:
            x: 3D-tensor, (batch_size, subsmp_len, attention_dim)
            pos_emb: 3D-tensor, (1, subsmp_len, attention_dim)
            mask_adder: 4D-tensor, (batch_size, 1, 1, subsmp_len)
        '''
        with tf.compat.v1.variable_scope(scope_name,
                                         dtype=self.dtype,
                                         use_resource=True) as scope:
            subsmp_len = (((self.config['maxlen_in'] - 3) // 2 + 1) - 3) // 2 + 1
            mask = tf.sequence_mask(input_len, maxlen=subsmp_len)
            mask = tf.cast(mask, scope.dtype)
            mask = tf.reshape(mask, (mask.shape[0], 1, 1, mask.shape[1]))
            mask_adder = (1.0 - mask) * self.mask_value

            # subsampling conv1, channels_last
            conv1 = tf.compat.v1.layers.Conv2D(self.config['adim'], 3, 2,
                                               activation="relu",
                                               use_bias=True,
                                               kernel_regularizer=self.kernel_regularizer,
                                               bias_regularizer=self.bias_regularizer,
                                               name='subsample/conv1')
            x = conv1(input)

            # subsampling conv2, channels_last
            conv2 = tf.compat.v1.layers.Conv2D(self.config['adim'], 3, 2,
                                               activation="relu",
                                               use_bias=True,
                                               kernel_regularizer=self.kernel_regularizer,
                                               bias_regularizer=self.bias_regularizer,
                                               name='subsample/conv2')
            x = conv2(x)
            x = tf.reshape(x, [x.shape[0], x.shape[1], -1])

            # embedding linear
            dense = tf.compat.v1.layers.Dense(units=self.config['adim'],
                                              use_bias=True,
                                              kernel_regularizer=self.kernel_regularizer,
                                              bias_regularizer=self.bias_regularizer,
                                              name="subsample/emb_ff")
            x = dense(x)

            # scaling
            x = math.sqrt(self.config['adim']) * x

            # position embedding
            _, length, dmodel = x.shape.as_list()
            pos_emb = self._build_pos_embedding(length, dmodel, reverse=True)

            if self.training:
                if self.config['use_ipu_dropout']:
                    wav_emb = ipu.rand_ops.dropout(x, rate=self.config['dropout_rate'])
                    pos_emb = ipu.rand_ops.dropout(pos_emb, rate=self.config['dropout_rate'])
                else:
                    wav_emb = tf.nn.dropout(x, rate=self.config['dropout_rate'])
                    pos_emb = tf.nn.dropout(pos_emb, rate=self.config['dropout_rate'])
            else:
                wav_emb = x

        return wav_emb, pos_emb, mask_adder

    def _build_layer_norm(self, input, scope_name):
        '''
        Args:
            input: 3D-tensor, (batch_size, length, attention_dim)
            scope_name: scope name
        Returns:
            x: layer normalized tensor, norm axis=-1
        '''
        with tf.compat.v1.variable_scope(scope_name,
                                         dtype=self.dtype,
                                         use_resource=True) as scope:
            x = ipu.normalization_ops.layer_norm(input, epsilon=1e-3,
                                                 training=self.training,
                                                 trainable=self.training,
                                                 scope="norm")
        return x

    def _build_feed_forward(self, input, scale, scope_name):
        '''
        Args:
            input: 3D-tensor, (batch_size, length, attention_dim)
            scope_name: scope name

        Returns:
            x: 3D-tensor, (batch_size, length, attention_dim)
        '''
        with tf.compat.v1.variable_scope(scope_name,
                                         dtype=self.dtype,
                                         use_resource=True) as scope:
            # linear 1
            dense_1 = tf.compat.v1.layers.Dense(units=self.config['eunits'],
                                                use_bias=True,
                                                kernel_regularizer=self.kernel_regularizer,
                                                bias_regularizer=self.bias_regularizer,
                                                name="ff/dense_1")
            x = swish(dense_1(input))

            if self.training:
                if self.config['use_ipu_dropout']:
                    x = ipu.rand_ops.dropout(x, rate=self.config['dropout_rate'])
                else:
                    x = tf.nn.dropout(x, rate=self.config['dropout_rate'])

            # linear 2
            dense_2 = tf.compat.v1.layers.Dense(units=self.config['adim'],
                                                use_bias=True,
                                                kernel_regularizer=self.kernel_regularizer,
                                                bias_regularizer=self.bias_regularizer,
                                                name="ff/dense_2")
            x = dense_2(x)

            if self.training:
                if self.config['use_ipu_dropout']:
                    x = ipu.rand_ops.dropout(x, rate=self.config['dropout_rate'])
                else:
                    x = tf.nn.dropout(x, rate=self.config['dropout_rate'])
            x = scale * x

        return x

    def _relative_shift(self, x):
        '''
        Args:
            x: 4D-tensor, (batch_size, n_head, length_q, length_v)

        Returns:
            4D-tensor, (batch_size, n_head, length_q, length_v)
        '''
        x_shape = tf.shape(x)
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
        x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[3] + 1, x_shape[2]])
        x = tf.reshape(x[:, :, 1:, :], x_shape)
        return x

    def _build_self_attention(self, query, key, value, scope_name,
                              mask_adder=None, pos_emb=None):
        '''
        Args:
            query: 3D-tensor, (batch_size, length_q, attention_dim)
            key: 3D-tensor, (batch_size, length_v, attention_dim)
            value: 3D-tensor, (batch_size, length_v, attention_dim)
            scope_name: scope name
            mask_adder: 4D-tensor, (batch_size, 1, 1, length_v) or
                                   (batch_size, 1, length_q, length_v)
            pos_emb: 3D-tensor, (1, length_q, attention_dim) or None

        Returns:
            3D-tensor, (batch_size, length_q, attention_dim)
        '''
        assert self.config['adim'] % self.config['aheads'] == 0
        head_size = self.config['adim'] // self.config['aheads']

        q_shape = (query.shape[0], query.shape[1], self.config['aheads'],
                   head_size)
        k_shape = (key.shape[0],   key.shape[1],   self.config['aheads'],
                   head_size)
        v_shape = k_shape
        input_shape = query.shape

        with tf.compat.v1.variable_scope(scope_name,
                                         dtype=self.dtype,
                                         use_resource=True) as scope:
            # qkv
            q_dense = tf.compat.v1.layers.Dense(units=self.config['adim'],
                                                use_bias=True,
                                                kernel_regularizer=self.kernel_regularizer,
                                                bias_regularizer=self.bias_regularizer,
                                                name="att/q_dense")
            k_dense = tf.compat.v1.layers.Dense(units=self.config['adim'],
                                                use_bias=True,
                                                kernel_regularizer=self.kernel_regularizer,
                                                bias_regularizer=self.bias_regularizer,
                                                name="att/k_dense")
            v_dense = tf.compat.v1.layers.Dense(units=self.config['adim'],
                                                use_bias=True,
                                                kernel_regularizer=self.kernel_regularizer,
                                                bias_regularizer=self.bias_regularizer,
                                                name="att/v_dense")

            # (batch_size, length, adim)
            query = q_dense(query)
            key = k_dense(key)
            value = v_dense(value)

            # (batch_size, n_head, length, head_size)
            query = tf.transpose(tf.reshape(query, q_shape), (0, 2, 1, 3))
            key = tf.transpose(tf.reshape(key,   k_shape), (0, 2, 1, 3))
            value = tf.transpose(tf.reshape(value, v_shape), (0, 2, 1, 3))

            # relative self-attention
            if pos_emb is not None:
                p_dense = tf.compat.v1.layers.Dense(units=self.config['adim'],
                                                    use_bias=False,
                                                    kernel_initializer=create_initializer(
                                                        self.config['initializer_range'], dtype=self.dtype),
                                                    kernel_regularizer=self.kernel_regularizer,
                                                    bias_regularizer=self.bias_regularizer,
                                                    name="att/p_dense")
                pos_bias_u = tf.compat.v1.get_variable(
                    'pos_bias_u', [self.config['aheads'], head_size],
                    scope.dtype,
                    initializer=tf.keras.initializers.glorot_uniform,
                    trainable=self.training,
                    regularizer=self.bias_regularizer
                )
                pos_bias_v = tf.compat.v1.get_variable(
                    'pos_bias_v', [self.config['aheads'], head_size],
                    scope.dtype,
                    initializer=tf.keras.initializers.glorot_uniform,
                    trainable=self.training,
                    regularizer=self.bias_regularizer
                )

                # (1, length_q, adim)
                pos = p_dense(pos_emb)
                # (1, length_q, n_head, head_size)
                pos = tf.reshape(
                    pos, (1, -1, self.config['aheads'], head_size))

                # (batch_size, length_q, n_head, head_size)
                query_with_u = tf.transpose(query, (0, 2, 1, 3)) + pos_bias_u
                query_with_v = tf.transpose(query, (0, 2, 1, 3)) + pos_bias_v

                # (batch_size, n_head, length_q, length_v)
                logits_with_u = tf.matmul(tf.transpose(query_with_u, (0, 2, 1, 3)),
                                          tf.transpose(key, (0, 1, 3, 2)))
                logits_with_v = tf.matmul(tf.transpose(query_with_v, (0, 2, 1, 3)),
                                          tf.transpose(pos, (0, 2, 3, 1)))
                logits_with_v = self._relative_shift(logits_with_v)

                logits = logits_with_u + logits_with_v
            else:
                logits = tf.matmul(query, tf.transpose(key, (0, 1, 3, 2)))

            # logits, (batch_size, n_head, length_q, length_v)
            logits = logits / math.sqrt(head_size)

            if mask_adder is not None:
                logits = tf.add(logits, mask_adder)

            scores = tf.nn.softmax(logits)
            # if mask_adder is not None:
            #    zeros = tf.zeros_like(mask_adder)
            #    scores = tf.multiply(scores, tf.where(mask_adder < 0, zeros, 1 - zeros))

            if self.training:
                if self.config['use_ipu_dropout']:
                    scores = ipu.rand_ops.dropout(scores, rate=self.config['attn_dropout_rate'])
                else:
                    scores = tf.nn.dropout(scores, rate=self.config['attn_dropout_rate'])

            # (batch_size, n_head, length_q, length_v) * (batch_size, n_head, length_v, head_size)
            #                                        |
            #                     (batch_size, n_head, length_q, head_size)
            qkv = tf.matmul(scores, value)

            # (batch_size, length_q, adim)
            qkv = tf.reshape(tf.transpose(qkv, (0, 2, 1, 3)), input_shape)

            # linear out
            o_dense = tf.compat.v1.layers.Dense(units=self.config['adim'],
                                                use_bias=True,
                                                kernel_regularizer=self.kernel_regularizer,
                                                bias_regularizer=self.bias_regularizer,
                                                name="att/o_dense")
            qkv_o = o_dense(qkv)

            if self.training:
                if self.config['use_ipu_dropout']:
                    qkv_o = ipu.rand_ops.dropout(qkv_o, rate=self.config['dropout_rate'])
                else:
                    qkv_o = tf.nn.dropout(qkv_o, rate=self.config['dropout_rate'])

            return qkv_o

    def _build_conv_module(self, input, scope_name):
        '''
        Args:
            input: 3D-tensor, (batch_size, length, attention_dim)
            scope_name: scope name

        Returns:
            3D-tensor, (batch_size, length, attention_dim)
        '''
        with tf.compat.v1.variable_scope(scope_name,
                                         dtype=self.dtype,
                                         use_resource=True) as scope:
            x = input

            # pointwise conv
            conv1 = tf.compat.v1.keras.layers.Conv1D(2 * self.config['adim'], 1, 1,
                                                     use_bias=True,
                                                     kernel_regularizer=self.kernel_regularizer,
                                                     bias_regularizer=self.bias_regularizer,
                                                     name='convolution/conv1',
                                                     dtype=self.dtype)
            x = conv1(x)
            x = glu(x[:, :, :self.config['adim']],
                    x[:, :, self.config['adim']:])

            # tf 1.15 don't support DepthWiseConv1D
            x = tf.expand_dims(x, axis=1)
            conv2 = tf.compat.v1.keras.layers.DepthwiseConv2D([1, self.config['kernel_size']],
                                                              use_bias=True,
                                                              padding='SAME',
                                                              depthwise_regularizer=self.kernel_regularizer,
                                                              bias_regularizer=self.bias_regularizer,
                                                              name='convolution/conv2',
                                                              dtype=self.dtype)
            x = tf.squeeze(conv2(x), axis=1)

            # replace `batch_normalization` with `layer_normalization`
            x = ipu.normalization_ops.layer_norm(x, epsilon=1e-3,
                                                 training=self.training,
                                                 trainable=self.training,
                                                 scope="norm")
            x = swish(x)

            # pointwise conv
            conv3 = tf.compat.v1.keras.layers.Conv1D(self.config['adim'], 1, 1,
                                                     padding='VALID',
                                                     kernel_regularizer=self.kernel_regularizer,
                                                     bias_regularizer=self.bias_regularizer,
                                                     name='convolution/conv3',
                                                     dtype=self.dtype)
            x = conv3(x)

            if self.training:
                if self.config['use_ipu_dropout']:
                    x = ipu.rand_ops.dropout(x, rate=self.config['dropout_rate'])
                else:
                    x = tf.nn.dropout(x, rate=self.config['dropout_rate'])
        return x

    def _build_encoder_layer(self, x, mask_adder, pos_emb, prefix):
        '''
        Args:
            x: 3D-tensor, (batch_size, length, attention_dim)
            mask_adder: 4D-tensor, (batch_size, 1, 1, length)
            pos_emb: 3D-tensor, (1, length, attention_dim)
            prefix: scope name prefix

        Returns:
            3D-tensor, (batch_size, length, attention_dim)
        '''
        scope_name = str(prefix)

        residual = x
        x = self._build_layer_norm(x, scope_name + '/norm_1')
        x = self._build_feed_forward(x, 0.5, scope_name + '/ff_1')
        x = x + residual

        residual = x
        x = self._build_layer_norm(x, scope_name + '/norm_2')
        x = self._build_self_attention(x, x, x, scope_name + '/self_att',
                                       mask_adder, pos_emb)
        x = x + residual

        residual = x
        x = self._build_layer_norm(x, scope_name + '/norm_3')
        x = self._build_conv_module(x, scope_name + '/conv_module')
        x = x + residual

        residual = x
        x = self._build_layer_norm(x, scope_name + '/norm_4')
        x = self._build_feed_forward(x, 0.5, scope_name + '/ff_2')
        x = x + residual

        x = self._build_layer_norm(x, scope_name + '/norm_5')

        return x

    def _build_encoder(self, x, pos_emb, mask_adder):
        '''
        Args:
            x: 3D-tensor, (batch_size, length, attention_dim)
            pos_emb: 3D-tensor, (1, length, attention_dim)
            mask_adder: 4D-tensor, (batch_size, 1, 1, length)

        Returns:
            3D-tensor, (batch_size, length, attention_dim)
        '''
        for i in range(self.config['elayers']):
            x = self._build_encoder_layer(x, mask_adder, pos_emb,
                                          "encoder/encoder_" + str(i))
        x = self._build_layer_norm(x, "encoder/norm")

        return x

    # decoder related
    def _build_decoder_embedding(self, input, seq_len, scope_name):
        '''
        Args:
            input: 2D-tensor, (batch_size, maxlen_tgt), target word index
            seq_len: 1D-tensor, (batch_size,), valid target word length
            scope_name: scope name

        Returns:
            x: 3D-tensor, (batch_size, maxlen_tgt, attention_dim)
            loss_mask: 3D-tensor, (batch_size, maxlen_tgt-1, 1)
            att_mask: 4D-tensor, (1, 1, maxlen_tgt-1, maxlen_tgt-1), constant
        '''
        with tf.compat.v1.variable_scope(scope_name,
                                         dtype=self.dtype,
                                         use_resource=True) as scope:
            loss_mask = tf.sequence_mask(seq_len,
                                         maxlen=self.config['maxlen_tgt'] - 1,
                                         dtype=scope.dtype)
            loss_mask = tf.expand_dims(loss_mask, axis=2)

            embedding = tf.compat.v1.get_variable(
                "embedding_table",
                [self.config['vocab_size'], self.config['adim']],
                scope.dtype,
                initializer=tf.initializers.random_uniform(
                    minval=0, maxval=1.0, dtype=scope.dtype
                ),
                trainable=self.training,
                regularizer=self.kernel_regularizer
            )

            # x = tf.nn.embedding_lookup(embedding, input)
            x = embedding_ops.embedding_lookup(embedding, input)

            # position embedding
            _, max_len, dmodel = x.shape.as_list()
            pos_emb = self._build_pos_embedding(max_len, dmodel, reverse=False)

            x = math.sqrt(self.config['adim']) * x + pos_emb

            if self.training:
                if self.config['use_ipu_dropout']:
                    x = ipu.rand_ops.dropout(x, rate=self.config['dropout_rate'])
                else:
                    x = tf.nn.dropout(x, rate=self.config['dropout_rate'])
            # subsequent_mask
            index = tf.range(1, self.config['maxlen_tgt'], 1, dtype=tf.int32)
            index = tf.reshape(index, (1, 1, -1))
            att_mask = tf.sequence_mask(index, dtype=scope.dtype)
            att_mask = (1.0 - att_mask) * self.mask_value

        return x, loss_mask, att_mask

    def _build_decoder_layer(self, tgt, tgt_mask, mem, mem_mask, prefix):
        '''
        Args:
            tgt: 3D-tensor, (batch_size, maxlen_tgt, attention_dim)
            tgt_mask: 4D-tensor, (batch_size, 1, maxlen_tgt, maxlen_tgt)
            mem: 3D-tensor, (batch_size, maxlen_wav, attention_dim)
            mem_mask: 4D-tensor, (batch_size, 1, 1, maxlen_wav)

        Returns:
            x: 3D-tensor, (batch_size, maxlen_tgt, attention_dim)
        '''
        scope_name = str(prefix)

        x = tgt

        residual = x
        x = self._build_layer_norm(x, scope_name + '/norm_1')
        x = self._build_self_attention(x, x, x,
                                       scope_name + '/tgt_att',
                                       tgt_mask)
        x = x + residual

        residual = x
        x = self._build_layer_norm(x, scope_name + '/norm_2')
        x = self._build_self_attention(x, mem, mem,
                                       scope_name + '/mem_att',
                                       mem_mask)
        x = x + residual

        residual = x
        x = self._build_layer_norm(x, scope_name + '/norm_3')
        x = self._build_feed_forward(x, 1.0, scope_name + '/ff')
        x = x + residual

        return x

    def _build_classifier_output(self, input, scope_name):
        '''
        Args:
            input: 3D-tensor, (batch_size, maxlen_tgt, attention_dim)
            scope_name: scope name

        Returns:
            3D-tensor, (batch_size, maxlen_tgt, vocab_size)
        '''
        with tf.compat.v1.variable_scope(scope_name,
                                         dtype=self.dtype,
                                         use_resource=True) as scope:
            dense = tf.compat.v1.layers.Dense(units=self.config['vocab_size'],
                                              use_bias=True,
                                              kernel_regularizer=self.kernel_regularizer,
                                              bias_regularizer=self.bias_regularizer,
                                              name="cls")
            x = dense(input)

            return x

    def _build_decoder(self, tgt, tgt_mask, mem, mem_mask):
        '''
        Args:
            tgt: 3D-tensor, (batch_size, maxlen_tgt, attention_dim)
            tgt_mask: 4D-tensor, (batch_size, 1, 1, maxlen_tgt)
            mem: 3D-tensor, (batch_size, maxlen_wav, attention_dim)
            mem_mask: 4D-tensor, (batch_size, 1, 1, maxlen_wav)

        Returns:
            3D-tensor, (batch_size, maxlen_tgt, vocab_size)
        '''
        x = tgt
        for i in range(self.config['dlayers']):
            x = self._build_decoder_layer(x, tgt_mask, mem, mem_mask,
                                          "decoder/decoder_" + str(i))

        x = self._build_layer_norm(x, "decoder/norm")
        x = self._build_classifier_output(x, "loss/kl_logits")

        return x


    def _build_kl_loss(self, logits, labels, mask):
        with tf.compat.v1.variable_scope("loss/kl_loss", use_resource=True):
            on_value = 1.0 - self.config['lsm_weight']
            off_value = self.config['lsm_weight'] / (self.config['vocab_size'] - 1)
            y_true = tf.one_hot(labels, self.config['vocab_size'],
                                on_value=on_value,
                                off_value=off_value)
            y_true = tf.cast(y_true, self.dtype)
            y_pred = tf.nn.log_softmax(logits)
            loss_pre = y_true * (tf.math.log(y_true) - y_pred) * mask
            loss = tf.reduce_sum(loss_pre)

        return loss

    def optimizer_function(self, lr, loss, kl_cls, tgt):
        optimizer_type = self.config['optimizer'].lower()
        loss = self.config['loss_scale'] * loss
        if optimizer_type == 'sgd':
            lr = lr / self.config['loss_scale']
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr)
        elif optimizer_type == 'sgdm':
            optimizer = tf.compat.v1.train.MomentumOptimizer(lr, 0.9)
        elif optimizer_type == 'adam':
            optimizer = tf.compat.v1.train.AdamOptimizer(lr,
                                                         beta1=0.9,
                                                         beta2=0.98,
                                                         epsilon=1e-6)
        elif optimizer_type == 'adaml':
            optimizer = AdamLossScalingOptimizer(lr, self.config['loss_scale'], weights_dtype=tf.float32)
        else:
            raise ValueError(f"Optimizer {optimizer_type} not implemented.")
        if self.config['replica'] > 1:
            optimizer = ipu.optimizers.cross_replica_optimizer.CrossReplicaOptimizer(optimizer)

        return pipelining_ops.OptimizerFunctionOutput(optimizer, loss)


    def _build_embedding_stage(self, input, input_len):
        enc_emb, pos_emb, enc_mask = \
            self._build_encoder_embedding(input, input_len, "encoder/embedding")
        return enc_emb, pos_emb, enc_mask

    def _build_encoder_stage(self, enc_emb, pos_emb, enc_mask):
        for i in range(self.tmp_start, self.tmp_end):
            enc_emb = self._build_encoder_layer(enc_emb, enc_mask, pos_emb,
                                                "encoder/encoder_" + str(i))
        return enc_emb, pos_emb, enc_mask

    def _build_decoder_embedding_stage(self, tgt, tgt_len):
        dec_emb, loss_mask, dec_mask = \
            self._build_decoder_embedding(tgt, tgt_len, "decoder/embedding")
        dec_emb = dec_emb[:, : -1, :]    # the first tgt is `sos`
        return tgt, dec_emb, loss_mask, dec_mask

    def _build_decoder_stage(self, enc_emb, enc_mask, dec_emb, dec_mask):
        for i in range(self.tmp_start, self.tmp_end):
            dec_emb = self._build_decoder_layer(
                dec_emb, dec_mask, enc_emb, enc_mask, "decoder/decoder_" + str(i))
        return enc_emb, enc_mask, dec_emb, dec_mask

    def _build_output_loss_stage(self, tgt, dec_emb, loss_mask):
        tgt_exclude_sos = tgt[:, 1:]
        kl_logits = self._build_layer_norm(dec_emb, "decoder/norm")
        kl_logits = self._build_classifier_output(kl_logits, "loss/kl_logits")
        kl_cls = tf.compat.v1.argmax(kl_logits, axis=-1, output_type=tf.int32)
        kl_loss = self._build_kl_loss(kl_logits, tgt_exclude_sos, loss_mask)
        loss = kl_loss / self.global_batch_size
        return loss, kl_cls, tgt_exclude_sos

    def _build_1st_stage(self, lr, input, input_len, tgt, tgt_len):
        enc_emb, pos_emb, enc_mask = self._build_embedding_stage(input, input_len)
        self.tmp_start, self.tmp_end = 0, 8
        enc_emb, pos_emb, enc_mask = self._build_encoder_stage(enc_emb, pos_emb, enc_mask)
        return lr, enc_emb, pos_emb, enc_mask, tgt, tgt_len

    def _build_2nd_stage(self, lr, enc_emb, pos_emb, enc_mask, tgt, tgt_len):
        self.tmp_start, self.tmp_end = 8, 16
        enc_emb, pos_emb, enc_mask = self._build_encoder_stage(enc_emb, pos_emb, enc_mask)
        tgt, dec_emb, loss_mask, dec_mask = self._build_decoder_embedding_stage(tgt, tgt_len)
        self.tmp_start, self.tmp_end = 0, 1
        enc_emb, enc_mask, dec_emb, dec_mask = self._build_decoder_stage(enc_emb, enc_mask, dec_emb, dec_mask)
        loss, kl_cls, tgt_exclude_sos = self._build_output_loss_stage(tgt, dec_emb, loss_mask)
        return lr, loss, kl_cls, tgt_exclude_sos

    def _build_computational_stages(self):
        self.computational_stages.append(
            partial(self._build_1st_stage)
        )
        self.computational_stages.append(
            partial(self._build_2nd_stage)
        )
        self.device_mapping = [0, 1]

    def get_global_batch_size(self):
        self.global_batch_size = self.config['gradient_accumulation_count'] * self.config['replica'] * self.config['batch_size']
        print('local batch size: {}, ga: {}, global batch size: {}'.format(self.config['batch_size'], self.config['gradient_accumulation_count'], self.global_batch_size))

    def run_with_pipeline(self):
        self._build_dataset()
        self._build_computational_stages()
        self.get_global_batch_size()
        self.outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

        def train(lr, infeed, outfeed, gradient_accumulation_count):
            pipeline_op = pipelining_ops.pipeline(
                self.computational_stages,
                gradient_accumulation_count=gradient_accumulation_count,
                gradient_accumulation_dtype=self.dtype,
                inputs=[lr],
                infeed_queue=infeed,
                outfeed_queue=outfeed,
                device_mapping=self.device_mapping,
                optimizer_function=self.optimizer_function,
                offload_weight_update_variables=False)

            return pipeline_op

        def infer(lr, infeed, outfeed, gradient_accumulation_count):
            pipeline_op = pipelining_ops.pipeline(
                self.computational_stages,
                gradient_accumulation_count=gradient_accumulation_count,
                gradient_accumulation_dtype=self.dtype,
                inputs=[lr],
                infeed_queue=infeed,
                outfeed_queue=outfeed,
                device_mapping=self.device_mapping)

            return pipeline_op

        model = train if self.training else infer
        with tf.compat.v1.device("cpu"):
            lr = tf.compat.v1.placeholder(np.float32, [])
        pipeline_md = partial(model,
                              lr=lr,
                              infeed=self.infeed_queue,
                              outfeed=self.outfeed_queue,
                              gradient_accumulation_count=self.config['gradient_accumulation_count'])

        with ipu_scope('/device:IPU:0'):
            compiled = ipu_compiler.compile(pipeline_md, [])
        outfeed = self.outfeed_queue.dequeue()
        saver = tf.compat.v1.train.Saver()
        total_parameters = 0
        variables = tf.compat.v1.trainable_variables()

        if not os.path.exists('logs'): os.mkdir('logs')
        with open('logs/' + self.config['logfile'], 'w') as fp:
            for var in variables:
                fp.write(str(var) + '\n')
                total_parameters += np.prod(var.shape)
            fp.write('\nTotal Parameters : ' + str(total_parameters) + '\n')

        # Create ipu_options
        # we assume one ipu for one stage here
        ipu_options = get_config(num_ipus=len(self.device_mapping) * self.config['replica'])
        ipu_options.configure_ipu_system()

        total_steps = self.data_loader.num_utts * self.config['epochs'] // self.global_batch_size
        print('total_steps: ', total_steps)
        if self.config['wandb_name'] is not None:
            try:
                import wandb
            except:
                raise ImportError('wandb not installed')
            wandb.init(self.config['wandb_name'])
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(self.infeed_queue.initializer)
            step_per_epoch = self.data_loader.num_utts // self.global_batch_size
            for epoch in range(1, self.config['epochs']+1):
                for step in range(1, step_per_epoch+1):
                    global_step = (epoch-1) * step_per_epoch + step
                    step_lr = self.get_lr(global_step)
                    start = time.time()
                    _ = sess.run(compiled, {lr: step_lr})
                    result = sess.run(outfeed)
                    duration = time.time() - start
                    if step % 10 == 0:
                        tput = self.global_batch_size / duration
                        print('epoch: {}/{}, global_step: {}/{}, loss: {}, through_put: {}'.format(epoch, self.config['epochs'], global_step, total_steps, np.mean(result[1]), tput))
                kl_acc = self.get_kl_acc(result[2], result[3])
                if self.config['wandb_name'] is not None:
                    wandb.log({
                        "loss": np.mean(result[1]),
                        'acc': kl_acc,
                    })
                if self.config['save_checkpoint']:
                    saver.save(sess, 'logs/model.ckpt', global_step=global_step)
            if self.config['freeze']:
                self.save_pb(sess, self.output_names)
