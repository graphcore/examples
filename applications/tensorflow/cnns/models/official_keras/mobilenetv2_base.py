# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow.compat.v1 as tf
from models import tf_layers as layers

ImageNetBlockType = Tuple[int, int, int, int]
CifarBlockType = Tuple[int, int, int]
TensorType = Union[tf.Tensor, np.ndarray, tf.placeholder]


# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2(object):
    """Base class for MobileNetV2 architecture."""

    def __init__(self, num_classes: int, image_width: int, image_height: int,
                 image_channels: Optional[int] = 3, weights: Optional[str] = 'imagenet',
                 alpha: Optional[int] = 1.0) -> None:

        """Initialize mobilenet model.

        Args:
            num_classes: Number of target classes.
            image_width: Input image width.
            image_height: Input image height.
            image_channels: Input image number of channels.
            weights: Inizialize with imagenet weights.
            alpha: controls network width
        """
        self.alpha = alpha
        self.num_classes = num_classes
        # Network is setup for NHWC
        self.input_shape = [image_height, image_width, image_channels]
        self.weights = weights

    def build_model(self, img_input: TensorType) -> TensorType:
        """Build graph using img_input as input.

                Args:
                    img_input: 4D Image input tensor of shape (batch, height, width, channels)

                Returns:
                    `Tensor` holding output probabilities per class, shape (batch, num_classes)
        """
        filters = _make_divisible(32 * self.alpha, 8)

        # Conv 1 block
        x = layers.zero_padding(img_input, padding=((0, 1), (0, 1)), name='Conv1_pad')
        x = layers.conv(x, filters_out=filters, kernel_size=3, padding='valid', add_bias=False, stride=2, name='Conv1')
        x = layers.norm(x, axis=-1, epsilon=1e-3, momentum=0.999, name='bn_Conv1')
        x = layers.relu(x, name='Conv1_relu', max_value=tf.constant(6, tf.float16))

        # Depthwise separable convolutions
        x = self._inverted_res_block(x, filters=16, alpha=self.alpha, stride=1,
                                     expansion=1, block_id=0)

        x = self._inverted_res_block(x, filters=24, alpha=self.alpha, stride=2,
                                     expansion=6, block_id=1)
        x = self._inverted_res_block(x, filters=24, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=2)

        x = self._inverted_res_block(x, filters=32, alpha=self.alpha, stride=2,
                                     expansion=6, block_id=3)
        x = self._inverted_res_block(x, filters=32, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=4)
        x = self._inverted_res_block(x, filters=32, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=5)

        x = self._inverted_res_block(x, filters=64, alpha=self.alpha, stride=2,
                                     expansion=6, block_id=6)
        x = self._inverted_res_block(x, filters=64, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=7)
        x = self._inverted_res_block(x, filters=64, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=8)
        x = self._inverted_res_block(x, filters=64, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=9)

        x = self._inverted_res_block(x, filters=96, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=10)
        x = self._inverted_res_block(x, filters=96, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=11)
        x = self._inverted_res_block(x, filters=96, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=12)

        x = self._inverted_res_block(x, filters=160, alpha=self.alpha, stride=2,
                                     expansion=6, block_id=13)
        x = self._inverted_res_block(x, filters=160, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=14)
        x = self._inverted_res_block(x, filters=160, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=15)

        x = self._inverted_res_block(x, filters=320, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=16)

        # no alpha applied to last conv as stated in the paper:
        # if the width multiplier is greater than 1 we
        # increase the number of output channels
        if self.alpha > 1.0:
            last_block_filters = _make_divisible(1280 * self.alpha, 8)
        else:
            last_block_filters = 1280

        x = layers.conv(x, filters_out=last_block_filters,
                        kernel_size=1,
                        add_bias=False,
                        name='Conv_1')
        x = layers.norm(x, epsilon=1e-3,
                        momentum=0.999,
                        name='Conv_1_bn')
        x = layers.relu(x, max_value=tf.constant(6, tf.float16), name='out_relu')

        # Include top
        x = layers.global_avg_pool(x)
        x = layers.fully_connected(x, self.num_classes, name='Logits')
        x = layers.softmax(x, name='act_softmax')
        return x

    def __call__(self, img_input: TensorType) -> TensorType:
        """Build graph using img_input as input."""
        return self.build_model(img_input)

    @staticmethod
    def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
        in_channels = inputs.get_shape()[-1]
        pointwise_conv_filters = int(filters * alpha)
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
        x = inputs
        prefix = 'block_{}_'.format(block_id)

        if block_id:
            # Expand
            x = layers.conv(x, filters_out=expansion * in_channels,
                            kernel_size=1,
                            padding='same',
                            add_bias=False,
                            name=prefix + 'expand')
            x = layers.norm(x, epsilon=1e-3,
                            momentum=0.999,
                            name=prefix + 'expand_BN')
            x = layers.relu(x, max_value=tf.constant(6, tf.float16), name=prefix + 'expand_relu')
        else:
            prefix = 'expanded_conv_'

        # Depthwise
        if stride == 2:
            x = layers.zero_padding(x, padding=((0, 1), (0, 1)),
                                    name=prefix + 'pad')
        x = layers.depthwise_conv(x, kernel_size=3,
                                  stride=stride,
                                  add_bias=False,
                                  padding='same' if stride == 1 else 'valid',
                                  name=prefix + 'depthwise')
        x = layers.norm(x, epsilon=1e-3,
                        momentum=0.999,
                        name=prefix + 'depthwise_BN')

        x = layers.relu(x, max_value=tf.constant(6, tf.float16), name=prefix + 'depthwise_relu')

        # Project
        x = layers.conv(x, filters_out=pointwise_filters,
                        kernel_size=1,
                        padding='same',
                        add_bias=False,
                        name=prefix + 'project')
        x = layers.norm(x,
                        epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')

        if in_channels == pointwise_filters and stride == 1:
            return x + inputs
        return x
