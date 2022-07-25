#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
# Copyright (c) 2019 YunYang1994 <dreameryangyun@sjtu.edu.cn>
# License: MIT (https://opensource.org/licenses/MIT)
# This file has been modified by Graphcore Ltd.

import core.common as common
import ipu_utils
import tensorflow as tf


def darknet53(trainable, use_gn, precision):

    funcs = []

    with tf.variable_scope('darknet'):
        wrapper = ipu_utils.convolutional(filters_shape=(3, 3,  3,  32), trainable=trainable,
                                          use_gn=use_gn, name='conv0', precision=precision)
        funcs.append(wrapper)
        wrapper = ipu_utils.convolutional(filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, use_gn=use_gn, name='conv1', downsample=True, precision=precision)
        funcs.append(wrapper)

        funcs += ipu_utils.residual_block(
            64,  32, 64, trainable=trainable, use_gn=use_gn, name='residual0', precision=precision)

        wrapper = ipu_utils.convolutional(filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, use_gn=use_gn, name='conv4', downsample=True, precision=precision)
        funcs.append(wrapper)

        for i in range(2):
            funcs += ipu_utils.residual_block(
                128,  64, 128, trainable=trainable, use_gn=use_gn, name=f'residual{i+1}', precision=precision)

        wrapper = ipu_utils.convolutional(filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, use_gn=use_gn, name='conv9', downsample=True, precision=precision)
        funcs.append(wrapper)

        for i in range(8):
            funcs += ipu_utils.residual_block(
                256, 128, 256, trainable=trainable, use_gn=use_gn, name=f'residual{i+3}', precision=precision)

        wrapper = ipu_utils.route(funcs.pop(), "route_1")
        funcs.append(wrapper)
        wrapper = ipu_utils.convolutional(filters_shape=(3, 3, 256, 512),
                                          trainable=trainable, use_gn=use_gn, name='conv26', downsample=True, precision=precision)
        funcs.append(wrapper)

        for i in range(8):
            funcs += ipu_utils.residual_block(
                512, 256, 512, trainable=trainable, use_gn=use_gn, name=f'residual{i+11}', precision=precision)

        wrapper = ipu_utils.route(funcs.pop(), "route_2")
        funcs.append(wrapper)
        wrapper = ipu_utils.convolutional(filters_shape=(3, 3, 512, 1024),
                                          trainable=trainable, use_gn=use_gn, name='conv43', downsample=True, precision=precision)
        funcs.append(wrapper)

        for i in range(4):
            funcs += ipu_utils.residual_block(
                1024, 512, 1024, trainable=trainable, use_gn=use_gn, name=f'residual{i+19}', precision=precision)

    return funcs
