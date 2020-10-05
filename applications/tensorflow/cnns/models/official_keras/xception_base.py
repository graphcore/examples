# Copyright 2019 Graphcore Ltd.
"""Xception v1 model."""

from typing import Optional, Tuple, Union

import numpy as np
import tensorflow.compat.v1 as tf

from models import tf_layers as layers

ImageNetBlockType = Tuple[int, int, int, int]
CifarBlockType = Tuple[int, int, int]
TensorType = Union[tf.Tensor, np.ndarray, tf.placeholder]


class Xception(object):
    """Base class for Xception architecture."""

    def __init__(self,
                 num_classes: int, image_width: Optional[int] = 299, image_height: Optional[int] = 299,
                 image_channels: Optional[int] = 3, weights: Optional[str] = 'imagenet') -> None:
        """Initialize xception model.

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

    def build_model(self, img_input: TensorType) -> TensorType:
        """Build graph using img_input as input.

        Args:
            img_input: 4D Image input tensor of shape (batch, height, width, channels)

        Returns:
            `Tensor` holding output probabilities per class, shape (batch, num_classes)
        """

        x = layers.conv(img_input, filters_out=32, kernel_size=3, stride=2, add_bias=False, name='block1_conv1')
        x = layers.norm(x, name='block1_conv1_bn')
        x = layers.relu(x, name='block1_conv1_act')
        x = layers.conv(x, filters_out=64, kernel_size=3, add_bias=False, name='block1_conv2')
        x = layers.norm(x, name='block1_conv2_bn')
        x = layers.relu(x, name='block1_conv2_act')

        residual = layers.conv(x, filters_out=128, kernel_size=1, stride=2, padding='same', add_bias=False)
        residual = layers.norm(residual, name="batch_normalization")

        x = layers.separable_conv(x, filters_out=128, kernel_size=3, padding='same', add_bias=False,
                                  name='block2_sepconv1')
        x = layers.norm(x, name='block2_sepconv1_bn')
        x = layers.relu(x, name='block2_sepconv2_act')
        x = layers.separable_conv(x, filters_out=128, kernel_size=3, padding='same', add_bias=False,
                                  name='block2_sepconv2')
        x = layers.norm(x, name='block2_sepconv2_bn')

        x = layers.max_pool(x, 3, strides=2, padding='same', name='block2_pool')
        x += residual

        residual = layers.conv(x, filters_out=256, kernel_size=1, stride=2, padding='same', add_bias=False)
        residual = layers.norm(residual, name="batch_normalization")

        x = layers.relu(x, name='block3_sepconv1_act')
        x = layers.separable_conv(x, filters_out=256, kernel_size=3,
                                  padding='same',
                                  add_bias=False,
                                  name='block3_sepconv1')
        x = layers.norm(x, name='block3_sepconv1_bn')
        x = layers.relu(x, name='block3_sepconv2_act')
        x = layers.separable_conv(x, filters_out=256, kernel_size=3,
                                  padding='same',
                                  add_bias=False,
                                  name='block3_sepconv2')
        x = layers.norm(x, name='block3_sepconv2_bn')

        x = layers.max_pool(x, 3, strides=2,
                            padding='same',
                            name='block3_pool')
        x += residual

        residual = layers.conv(x, filters_out=728, kernel_size=1,
                               stride=2,
                               padding='same',
                               add_bias=False)
        residual = layers.norm(residual, name="batch_normalization")

        x = layers.relu(x, name='block4_sepconv1_act')
        x = layers.separable_conv(x, filters_out=728, kernel_size=3,
                                  padding='same',
                                  add_bias=False,
                                  name='block4_sepconv1')
        x = layers.norm(x, name='block4_sepconv1_bn')
        x = layers.relu(x, name='block4_sepconv2_act')
        x = layers.separable_conv(x, filters_out=728, kernel_size=3,
                                  padding='same',
                                  add_bias=False,
                                  name='block4_sepconv2')
        x = layers.norm(x, name='block4_sepconv2_bn')

        x = layers.max_pool(x, 3, strides=2,
                            padding='same',
                            name='block4_pool')
        x += residual

        for i in range(8):
            residual = x
            prefix = 'block' + str(i + 5)

            x = layers.relu(x, name=prefix + '_sepconv1_act')
            x = layers.separable_conv(x, filters_out=728, kernel_size=3,
                                      padding='same',
                                      add_bias=False,
                                      name=prefix + '_sepconv1')
            x = layers.norm(x, name=prefix + '_sepconv1_bn')
            x = layers.relu(x, name=prefix + '_sepconv2_act')
            x = layers.separable_conv(x, filters_out=728, kernel_size=3,
                                      padding='same',
                                      add_bias=False,
                                      name=prefix + '_sepconv2')
            x = layers.norm(x, name=prefix + '_sepconv2_bn')
            x = layers.relu(x, name=prefix + '_sepconv3_act')
            x = layers.separable_conv(x, filters_out=728, kernel_size=3,
                                      padding='same',
                                      add_bias=False,
                                      name=prefix + '_sepconv3')
            x = layers.norm(x, name=prefix + '_sepconv3_bn')

            x += residual

        residual = layers.conv(x, filters_out=1024, kernel_size=1, stride=2,
                               padding='same', add_bias=False)
        residual = layers.norm(residual, name="batch_normalization")

        x = layers.relu(x, name='block13_sepconv1_act')
        x = layers.separable_conv(x, filters_out=728, kernel_size=3,
                                  padding='same',
                                  add_bias=False,
                                  name='block13_sepconv1')
        x = layers.norm(x, name='block13_sepconv1_bn')
        x = layers.relu(x, name='block13_sepconv2_act')
        x = layers.separable_conv(x, filters_out=1024, kernel_size=3,
                                  padding='same',
                                  add_bias=False,
                                  name='block13_sepconv2')
        x = layers.norm(x, name='block13_sepconv2_bn')

        x = layers.max_pool(x, 3,
                            strides=2,
                            padding='same',
                            name='block13_pool')
        x += residual

        x = layers.separable_conv(x, filters_out=1536, kernel_size=3,
                                  padding='same',
                                  add_bias=False,
                                  name='block14_sepconv1')
        x = layers.norm(x, name='block14_sepconv1_bn')
        x = layers.relu(x, name='block14_sepconv1_act')

        x = layers.separable_conv(x, filters_out=2048, kernel_size=3,
                                  padding='same',
                                  add_bias=False,
                                  name='block14_sepconv2')
        x = layers.norm(x, name='block14_sepconv2_bn')
        x = layers.relu(x, name='block14_sepconv2_act')

        # Classification block
        x = layers.avg_pool(x, kernel_size=10, strides=1, name='avg_pool')
        x = layers.squeeze(x, axis=[1, 2], name='squeeze')
        x = layers.fully_connected(x, self.num_classes, name='predictions')
        x = layers.softmax(x, name='output-prob')

        return x

    def __call__(self, img_input: TensorType) -> TensorType:
        """Build graph using img_input as input.
        """
        return self.build_model(img_input)
