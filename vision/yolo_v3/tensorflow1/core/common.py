#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
# Copyright (c) 2019 YunYang1994 <dreameryangyun@sjtu.edu.cn>
# License: MIT (https://opensource.org/licenses/MIT)
# This file has been modified by Graphcore Ltd.

import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.ops import array_ops, linalg_ops, math_ops


def group_norm(input_tensor, groups=8, name='GroupNorm', trainable=True):
    """Run layer normalization on the last dimension of the tensor."""
    input_tensor = ipu.normalization_ops.group_norm(input_tensor, groups=groups, center=True, scale=True, epsilon=0.001,
                                                    training=True,
                                                    channels_axis=-1, scope=name)

    return input_tensor


def convolutional(input_data,
                  filters_shape,
                  trainable,
                  use_gn,
                  name,
                  precision=tf.float16,
                  downsample=False,
                  activate=True,
                  norm=True,
                  weight_centering=False):

    # this can reduce some memory usage
    @ipu.outlined_function
    def func(input_data):
        with tf.variable_scope(name):
            if downsample:
                pad_h = (filters_shape[0] - 2) // 2 + 1
                pad_w = (filters_shape[1] - 2) // 2 + 1
                paddings = tf.constant(
                    [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
                input_data = tf.pad(input_data, paddings, 'CONSTANT')
                strides = (1, 2, 2, 1)
                padding = 'VALID'
            else:
                strides = (1, 1, 1, 1)
                padding = "SAME"

            weight = tf.get_variable(name='weight', dtype=precision, trainable=True,
                                     shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))

            conv = tf.nn.conv2d(input=input_data, filter=weight,
                                strides=strides, padding=padding)
            if norm:
                if use_gn:
                    conv = group_norm(conv)
                else:
                    conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                         gamma_initializer=tf.ones_initializer(),
                                                         moving_mean_initializer=tf.zeros_initializer(),
                                                         moving_variance_initializer=tf.ones_initializer(), training=trainable)

            else:

                bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                       dtype=precision, initializer=tf.constant_initializer(0.0))

                conv = tf.nn.bias_add(conv, bias)

            if activate:
                conv = tf.nn.leaky_relu(conv, alpha=0.1)
        return conv
    return func(input_data)


def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, use_gn, name, precision):

    x = input_data

    with tf.variable_scope(name):
        x = convolutional(x, filters_shape=(1, 1, input_channel, filter_num1),
                          trainable=trainable, use_gn=use_gn, name='conv1', precision=precision)
        x = convolutional(x, filters_shape=(3, 3, filter_num1,   filter_num2),
                          trainable=trainable, use_gn=use_gn, name='conv2', precision=precision)

        residual_output = input_data + x

    return residual_output


def upsample(input_data, name, method="deconv", precision=tf.float16):
    assert method in ["resize", "deconv"]

    if method == "resize":
        # IPU don't have a resize_nearest_neighbor implemented currently
        # so we use stack and reshape to make one
        shape = tf.shape(input_data)
        # we repeat input feature map in channel dimension
        input_data = tf.stack([input_data, input_data], axis=3)
        # we reshape the stacked result to make every point repeat for 2 times in width dimension
        input_data = tf.reshape(input_data, shape=[shape[0], shape[1], shape[2]*2, shape[3]])
        # we stack the result again in width dimension
        input_data = tf.stack([input_data, input_data], axis=2)
        # after this reshape, every row will repeat for 2 times in height dimension
        # now we can get a upsample
        output = tf.reshape(input_data, shape=[shape[0], shape[1]*2, shape[2]*2, shape[3]])

    if method == "deconv":
        # with deconv, we can get a similar mAP as nearest_neighbor upsample
        num_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, num_filter, kernel_size=2, padding='same',
                                            strides=(2, 2), kernel_initializer=tf.random_normal_initializer(), dtype=precision)
    return output
