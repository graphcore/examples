# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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
Various helper functions for model definition.
"""

import tensorflow as tf
from tensorflow.python.ipu import normalization_ops

DTYPE = tf.float16


def _get_variable(name: str, shape: list, init):
    """
    Return the value of a given variable

    :param name: (str) prescribed variable name
    :param shape: (list) tensor shape
    :param init: (function) initialization function for given session
    :return:
    """
    return tf.get_variable(name, shape, initializer=init, dtype=DTYPE)


def relu(x: tf.Tensor) -> tf.Tensor:
    """
    ReLU activation

    :param x: (tf.Tensor) input tensor
    :return:
        (tf.Tensor) resulting activation function
    """
    return tf.nn.relu(x)


def maxpool(x: tf.Tensor, size=3, stride=2) -> tf.Tensor:
    """
    Maxpooling evaluation

    :param x: (tf.Tensor) input tensor
    :param size:  (int) size of the pooling layer
    :param stride: (int) stride of pooling layer
    :return:
        (tf.Tensor) pooled resulting tensor
    """
    x = tf.nn.max_pool(
        x,
        ksize=[1, size, size, 1],
        strides=[1, stride, stride, 1],
        padding='SAME')
    return x


def fc(x: tf.Tensor, num_units_out: int) -> tf.Tensor:
    """
    Fully-connected layer

    :param x: (tf.Tensor) input tensor
    :param num_units_out: (int) number of nodes
    :return:
        output of evaluating fc layer
    """
    num_units_in = x.get_shape()[1]
    w_init = tf.contrib.layers.xavier_initializer(dtype=DTYPE)
    b_init = tf.constant_initializer(0.0)

    weights = _get_variable('weights', shape=[num_units_in, num_units_out], init=w_init)
    biases = _get_variable('biases', shape=[num_units_out], init=b_init)

    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def reduce_mean(x: tf.Tensor, indices=(1, 2)) -> tf.Tensor:
    """
    Mean reduction

    :param x: (tf.Tensor) input tensor
    :param indices:  (tuple) indices of the reduction
    :return:
        (tf.Tensor) result of mean reduction
    """
    x = tf.reduce_mean(x, reduction_indices=indices)
    return x


def norm(x: tf.Tensor, norm_type='BATCH', groups=32, training=False) -> tf.Tensor:
    """
    Batch-normalization

    :param x: (tf.Tensor) input tensor
    :param norm_type: (str) type of normalization
    :param groups: (int) size of group
    :param training: (boolean) training flag
    :return:
    """
    if norm_type == 'BATCH':
        # Perhaps use tf.nn.fused_batch_norm instead.
        x = tf.layers.batch_normalization(x, fused=True, center=True, scale=True,
                                          training=training, trainable=training,
                                          momentum=0.997, epsilon=1e-5)
    elif norm_type == 'GROUP':
        x = normalization_ops.group_norm(x, groups=groups, center=True, scale=True,
                                         training=training, trainable=training,
                                         channels_axis=-1, reduction_axes=[-3, -2])
    return x


def conv(x: tf.Tensor, ksize=7, stride=1, filters_out=64, bias=True, name="conv_1",
         dilation_rate=(1, 1), padding='SAME') -> tf.Tensor:
    """
    Convolutional layer

    :param x: (tf.Tensor) input tensor
    :param ksize: (int) kernel size
    :param stride: (int) stride
    :param filters_out: (int) depth of output
    :param bias: (boolean) flag for bias evaluation
    :param name: (str) name of layer
    :param dilation_rate: (tuple) dilation parameters
    :param padding: (str) type of padding -- 'SAME' or 'VALID'
    :return:
        (tf.Tensor) resulting output tensor
    """
    filters_in = x.get_shape()[-1]

    wshape = [ksize, ksize, filters_in, filters_out]
    w_init = tf.contrib.layers.xavier_initializer(dtype=DTYPE)
    weights = _get_variable('weights' + name, shape=wshape, init=w_init)
    x = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding, dilations=dilation_rate)
    if bias:
        bshape = [filters_out]
        b_init = tf.zeros_initializer()
        biases = _get_variable('biases' + name, shape=bshape, init=b_init)
        x = x + biases
    return x
