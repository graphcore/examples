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
"""Create InceptionV1 graph.

Follows naming convention in
https://github.com/tensorflow/models/blob/61f8185dcd04a3611f3818fca766c8dc416a2b7b/research/slim/nets/inception_v1.py
"""
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow.compat.v1 as tf
from models.tf_layers import conv_norm_relu, max_pool, concat, avg_pool, \
    conv, squeeze, softmax

TensorType = Union[tf.Tensor, np.ndarray, tf.placeholder]


class InceptionV1(object):
    """Base class for InceptionV1 architecture."""

    def __init__(self,
                 num_classes: int, image_width: Optional[int] = 224,
                 image_height: Optional[int] = 224,
                 image_channels: Optional[int] = 3,
                 weights: Optional[str] = 'imagenet') -> None:
        """Initialize InceptionV1 model.

        Args:
            num_classes: Number of target classes.
            image_width: Input image width.
            image_height: Input image height.
            image_channels: Input image number of channels.
            weights: Initialize with imagenet weights.

        """

        self.num_classes = num_classes
        # Network is set up for NHWC
        self.input_shape = [image_height, image_width, image_channels]
        self.weights = weights

    @staticmethod
    def inception_block(x: TensorType, specs: Tuple, channel_axis: int,
                        name: str,
                        weight_suffix: Optional[str] = "weights",
                        conv_suffix: Optional[str] = "",
                        norm_suffix: Optional[str] = "/BatchNorm") -> TensorType:
        """Inception block.

        Args:
            x: input_tensor
            specs: Number of filters per branch.
            channel_axis: channel dimension
            name: Prefix for ths block.
            weight_suffix: Name of learn-able parameters in conv.
            conv_suffix: Suffix for conv layer.
            norm_suffix: Suffix for batch norm.

        Returns: Concatenated output of inception block.

        """
        (br0, br1, br2, br3) = specs  # ((64,), (96,128), (16,32), (32,))

        branch_0 = conv_norm_relu(x, br0[0], 1, 1,
                                  name=name + "/Branch_0/Conv2d_0a_1x1",
                                  weight_suffix=weight_suffix,
                                  conv_suffix=conv_suffix,
                                  norm_suffix=norm_suffix)

        branch_1 = conv_norm_relu(x, br1[0], 1, 1,
                                  name=name + "/Branch_1/Conv2d_0a_1x1",
                                  weight_suffix=weight_suffix,
                                  conv_suffix=conv_suffix,
                                  norm_suffix=norm_suffix)
        branch_1 = conv_norm_relu(branch_1, br1[1], 3, 3,
                                  name=name + "/Branch_1/Conv2d_0b_3x3",
                                  weight_suffix=weight_suffix,
                                  conv_suffix=conv_suffix,
                                  norm_suffix=norm_suffix)

        branch_2 = conv_norm_relu(x, br2[0], 1, 1,
                                  name=name + "/Branch_2/Conv2d_0a_1x1",
                                  weight_suffix=weight_suffix,
                                  conv_suffix=conv_suffix,
                                  norm_suffix=norm_suffix)
        if name == "InceptionV1/Mixed_5b":
            branch_2b_name = name + "/Branch_2/Conv2d_0a_3x3"
        else:
            branch_2b_name = name + "/Branch_2/Conv2d_0b_3x3"
        branch_2 = conv_norm_relu(branch_2, br2[1], 3, 3,
                                  name=branch_2b_name,
                                  weight_suffix=weight_suffix,
                                  conv_suffix=conv_suffix,
                                  norm_suffix=norm_suffix)

        branch_3 = max_pool(x, 3, strides=1, padding='same',
                            name=name + "/Branch_3/Conv2d_0a_max")
        branch_3 = conv_norm_relu(branch_3, br3[0], 1, 1,
                                  name=name + "/Branch_3/Conv2d_0b_1x1",
                                  weight_suffix=weight_suffix,
                                  conv_suffix=conv_suffix,
                                  norm_suffix=norm_suffix)

        x = concat(
            [branch_0, branch_1, branch_2, branch_3],
            axis=channel_axis,
            name=name + "_Concatenated")
        return x

    def build_model(self, img_input: TensorType) -> TensorType:
        """Build graph using img_input as input.

        Args:
            img_input: 4D Image input tensor of shape (batch, height, width,
            channels)

        Returns:
            `Tensor` holding output probabilities per class, shape (batch,
            num_classes)
        """
        channel_axis = -1

        x = conv_norm_relu(img_input, 64, 7, strides=2, padding='SAME',
                           name='InceptionV1/Conv2d_1a_7x7',
                           norm_suffix="/BatchNorm", weight_suffix="weights",
                           conv_suffix="")

        x = max_pool(x, 3, strides=2, padding='same', name='MaxPool_2a_3x3')
        x = conv_norm_relu(x, 64, 1, padding='same',
                           name='InceptionV1/Conv2d_2b_1x1',
                           weight_suffix="weights",
                           conv_suffix="", norm_suffix="/BatchNorm")
        x = conv_norm_relu(x, 192, 3, padding='same',
                           name='InceptionV1/Conv2d_2c_3x3',
                           weight_suffix="weights",
                           conv_suffix="", norm_suffix="/BatchNorm")
        x = max_pool(x, 3, strides=2, padding='same', name='MaxPool_3a_3x3')

        # Now the '3' level inception units
        x = self.inception_block(x, ((64,), (96, 128), (16, 32), (32,)),
                                 channel_axis, 'InceptionV1/Mixed_3b')
        x = self.inception_block(x, ((128,), (128, 192), (32, 96), (64,)),
                                 channel_axis, 'InceptionV1/Mixed_3c')

        x = max_pool(x, 3, strides=2, padding='same', name='MaxPool_4a_3x3')

        # Now the '4' level inception units
        x = self.inception_block(x, ((192,), (96, 208), (16, 48), (64,)),
                                 channel_axis, 'InceptionV1/Mixed_4b')
        x = self.inception_block(x, ((160,), (112, 224), (24, 64), (64,)),
                                 channel_axis, 'InceptionV1/Mixed_4c')
        x = self.inception_block(x, ((128,), (128, 256), (24, 64), (64,)),
                                 channel_axis, 'InceptionV1/Mixed_4d')
        x = self.inception_block(x, ((112,), (144, 288), (32, 64), (64,)),
                                 channel_axis, 'InceptionV1/Mixed_4e')
        x = self.inception_block(x, ((256,), (160, 320), (32, 128), (128,)),
                                 channel_axis, 'InceptionV1/Mixed_4f')

        x = max_pool(x, 2, strides=2, padding='same', name='MaxPool_5a_2x2')

        # Now the '5' level inception units
        x = self.inception_block(x, ((256,), (160, 320), (32, 128), (128,)),
                                 channel_axis, 'InceptionV1/Mixed_5b')
        x = self.inception_block(x, ((384,), (192, 384), (48, 128), (128,)),
                                 channel_axis, 'InceptionV1/Mixed_5c')

        # Classification block
        x = avg_pool(x, kernel_size=7, strides=1, name='avg_pool',
                     padding='valid')
        x = conv(x, filters_out=self.num_classes + 1, kernel_size=1,
                 padding='valid', add_bias=True,
                 name='InceptionV1/Logits/Conv2d_0c_1x1',
                 weight_suffix="weights", bias_suffix="biases")
        x = squeeze(x, axis=[1, 2], name='squeeze')
        x = softmax(x, name='output-prob')

        return x

    def __call__(self, img_input: TensorType) -> TensorType:
        """Build graph using img_input as input.
        """
        return self.build_model(img_input)
