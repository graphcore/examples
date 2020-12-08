# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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

from models.resnet_base import ResNet

import tensorflow.compat.v1 as tf
import tensorflow.contrib as contrib
from tensorflow.python.ipu import normalization_ops

# This is all written for: NHWC


class TensorflowResNet(ResNet):
    def __init__(self, *args, **kwargs):
        self.dtype = tf.float16
        super(TensorflowResNet, self).__init__(*args, **kwargs)

    def _get_variable(self, name, shape, init):
        return tf.get_variable(name, shape, initializer=init, dtype=self.dtype)

    def residual(self, x, shortcut, out_filters, stride, type='B'):
        in_shape = shortcut.get_shape()
        pad = int(x.get_shape()[3] - in_shape[3])
        if pad != 0 or type == 'C':
            if type == 'A':
                shortcut = tf.strided_slice(shortcut, [0, 0, 0, 0], in_shape,
                                            strides=[1, stride, stride, 1])
                shortcut = tf.pad(shortcut, paddings=[[0, 0], [0, 0], [0, 0], [0, pad]])
            else:
                shortcut = self.conv(shortcut, 1, stride, out_filters)
                shortcut = self.norm(shortcut)
        x = shortcut + x
        x = self.relu(x)
        return x

    def relu(self, x):
        return tf.nn.relu(x)

    def conv(self, x, ksize, stride, filters_out, bias=True):
        filters_in = x.get_shape()[-1]

        wshape = [ksize, ksize, filters_in, filters_out]
        w_init = contrib.layers.xavier_initializer(dtype=self.dtype)
        weights = self._get_variable('weights', shape=wshape, init=w_init)
        x = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
        if bias:
            bshape = [filters_out]
            b_init = tf.zeros_initializer()
            biases = self._get_variable('biases', shape=bshape, init=b_init)
            x = x + biases
        return x

    def norm(self, x, type='BATCH', groups=32, training=False):
        if type == 'BATCH':
            # Perhaps use tf.nn.fused_batch_norm instead.
            x = tf.layers.batch_normalization(x, fused=True, center=True, scale=True,
                                              training=training, trainable=training,
                                              momentum=0.997, epsilon=1e-5)
        elif type == 'GROUP':
            x = normalization_ops.group_norm(x, groups=groups, center=True, scale=True,
                                             training=training, trainable=training,
                                             channels_axis=-1, reduction_axes=[-3, -2])
        return x

    def fc(self, x, num_units_out):
        num_units_in = x.get_shape()[1]
        w_init = contrib.layers.xavier_initializer(dtype=self.dtype)
        b_init = tf.constant_initializer(0.0)

        with self.namescope('fc'):
            weights = self._get_variable('weights', shape=[num_units_in, num_units_out], init=w_init)
            biases = self._get_variable('biases', shape=[num_units_out], init=b_init)

            x = tf.nn.xw_plus_b(x, weights, biases)
        return x

    def reduce_mean(self, x, indices=(1, 2)):
        x = tf.reduce_mean(x, reduction_indices=indices)
        return x

    def maxpool(self, x):
        x = tf.nn.max_pool(
            x,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding='SAME')
        return x

    def namescope(self, debug_string):
        return tf.variable_scope(debug_string)
