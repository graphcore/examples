# Copyright 2020 Graphcore Ltd.
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from models import tf_layers as layers

ImageNetBlockType = Tuple[int, int, int, int]
CifarBlockType = Tuple[int, int, int]
TensorType = Union[tf.Tensor, np.ndarray, tf.placeholder]


class MobileNet(object):
    """Base class for MobileNet architecture."""

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
        filters = int(32 * self.alpha)
        shape = (-1, 1, 1, int(1024 * self.alpha))

        # Conv 1 block
        x = layers.zero_padding(img_input, padding=((0, 1), (0, 1)), name='conv1_pad')
        x = layers.conv(x, filters_out=filters, kernel_size=3, padding='valid', add_bias=False, stride=2, name='conv1')
        x = layers.norm(x, axis=-1, name='conv1_bn')
        x = layers.relu(x, name='conv1_relu')

        # Depthwise convolutions
        x = self._depthwise_conv_block(x, 64, self.alpha, depth_multiplier=1, block_id=1)
        x = self._depthwise_conv_block(x, 128, self.alpha, depth_multiplier=1, strides=2, block_id=2)
        x = self._depthwise_conv_block(x, 128, self.alpha, depth_multiplier=1, block_id=3)
        x = self._depthwise_conv_block(x, 256, self.alpha, depth_multiplier=1, strides=2, block_id=4)
        x = self._depthwise_conv_block(x, 256, self.alpha, depth_multiplier=1, block_id=5)
        x = self._depthwise_conv_block(x, 512, self.alpha, depth_multiplier=1, strides=2, block_id=6)
        x = self._depthwise_conv_block(x, 512, self.alpha, depth_multiplier=1, block_id=7)
        x = self._depthwise_conv_block(x, 512, self.alpha, depth_multiplier=1, block_id=8)
        x = self._depthwise_conv_block(x, 512, self.alpha, depth_multiplier=1, block_id=9)
        x = self._depthwise_conv_block(x, 512, self.alpha, depth_multiplier=1, block_id=10)
        x = self._depthwise_conv_block(x, 512, self.alpha, depth_multiplier=1, block_id=11)
        x = self._depthwise_conv_block(x, 1024, self.alpha, depth_multiplier=1, strides=2, block_id=12)
        x = self._depthwise_conv_block(x, 1024, self.alpha, depth_multiplier=1, block_id=13)

        # Include top
        x = layers.global_avg_pool(x)
        x = layers.reshape(x, shape=shape, name='reshape_1')
        x = layers.conv(x, filters_out=self.num_classes, kernel_size=1, padding='same', name='conv_preds',
                        add_bias=False)
        x = layers.reshape(x, shape=(-1, self.num_classes), name='reshape_2')
        x = layers.softmax(x, name='act_softmax')
        return x

    def __call__(self, img_input: TensorType) -> TensorType:
        """Build graph using img_input as input."""
        return self.build_model(img_input)

    @staticmethod
    def _depthwise_conv_block(x: Union[tf.Tensor, np.ndarray], pointwise_conv_filters, alpha, depth_multiplier=1,
                              strides=1, block_id=1) -> TensorType:
        """A block of depthwise convolutions

        Args:
            x: input tensor.
            pointwise_conv_filters: number of pointwise filters modified by the size param Alpha
            alpha: determines the network size (modifies the number of conv filters)
            depth_multiplier: Changes the number of depthwise filters
            strides: changes stride
            block_id: Used to identify blocks

        Returns:
            output tensor for the block.
        """

        pointwise_conv_filters = (pointwise_conv_filters * alpha)

        if strides == 1:
            _x = x
        else:
            _x = layers.zero_padding(x, padding=((0, 1), (0, 1)), name='conv_pad_%d' % block_id)

        _x = layers.depthwise_conv(_x, kernel_size=3, padding='same' if strides == 1 else 'valid',
                                   filters_out=depth_multiplier, stride=strides, add_bias=False,
                                   name='conv_dw_%d' % block_id)
        _x = layers.norm(_x, axis=-1, name='conv_dw_%d_bn' % block_id)

        _x = layers.relu(_x, name='conv_dw_%d' % block_id)

        _x = layers.conv(_x, filters_out=pointwise_conv_filters, kernel_size=1, padding='same',
                         add_bias=False, stride=1, name='conv_pw_%d' % block_id)
        _x = layers.norm(_x, axis=-1, name='conv_pw_%d_bn' % block_id)
        return layers.relu(_x, name='conv_pw_%d_relu' % block_id)
