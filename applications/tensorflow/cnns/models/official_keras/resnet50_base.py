# Copyright 2019 Graphcore Ltd.
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from models import tf_layers as layers

ImageNetBlockType = Tuple[int, int, int, int]
CifarBlockType = Tuple[int, int, int]
TensorType = Union[tf.Tensor, np.ndarray, tf.placeholder]


class ResNet50(object):
    """Base class for Resnet50 architecture."""

    def __init__(self,
                 num_classes: int, image_width: int, image_height: int,
                 image_channels: Optional[int] = 3, weights: Optional[str] = 'imagenet') -> None:
        """Initialize Resnet50 model.

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

        x = layers.conv(img_input, filters_out=64, kernel_size=7, stride=2, add_bias=False, name='conv1_conv')
        x = layers.norm(x, axis=-1, epsilon=1.001e-5, name='conv1_bn')
        x = layers.relu(x, name='conv1_relu')
        x = layers.zero_padding(x, padding=((1, 1), (1, 1)), name='pool1_pad')
        x = layers.max_pool(x, kernel_size=3, name='pool1')

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='1', strides=1)
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='2')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='3')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='1')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='2')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='3')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='4')

        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='1')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='2')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='3')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='4')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='5')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='6')

        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='1')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='2')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='3')

        x = layers.avg_pool(x, kernel_size=7, strides=1, name='avg_pool')
        x = layers.squeeze(x, axis=[1, 2], name='squeeze')
        x = layers.fully_connected(x, self.num_classes, name='probs')
        x = layers.softmax(x, name='output-prob')
        return x

    def __call__(self, img_input: TensorType) -> TensorType:
        """Build graph using img_input as input.
        """
        return self.build_model(img_input)

    @staticmethod
    def conv_block(input_tensor: Union[tf.Tensor, np.ndarray], kernel_size, filters,
                   stage,
                   block,
                   strides=2) -> TensorType:
        """Building block for a dense block.

        Args:
            input_tensor: Input tensor of type tf.Tensor if using tf backend, np.ndarray if using popart builder.
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the first conv layer in the block.

        Return:
            Output tensor for the block.
        """

        name_base = f"conv{stage}_block{block}_"

        shortcut = layers.conv(input_tensor, filters_out=filters[2], kernel_size=1, add_bias=False, stride=strides,
                               name=name_base + '0_conv')
        shortcut = layers.norm(shortcut, axis=-1, epsilon=1.001e-5, name=name_base + '0_bn')

        x = layers.conv(input_tensor, filters_out=filters[0], kernel_size=1, stride=strides,
                        add_bias=False, name=name_base + '1_conv')
        x = layers.norm(x, axis=-1, epsilon=1.001e-5, name=name_base + '1_bn')
        x = layers.relu(x, name=name_base + '1_relu')

        x = layers.conv(x, filters_out=filters[1], kernel_size=kernel_size, add_bias=False,
                        name=name_base + '2_conv')
        x = layers.norm(x, axis=-1, epsilon=1.001e-5, name=name_base + '2_bn')
        x = layers.relu(x, name=name_base + '2_relu')

        x = layers.conv(x, filters_out=filters[2], kernel_size=1, add_bias=False,
                        name=name_base + '3_conv')
        x = layers.norm(x, axis=-1, epsilon=1.001e-5, name=name_base + '3_bn')

        x = x + shortcut
        x = layers.relu(x, name=name_base + '3_relu')

        return x

    @staticmethod
    def identity_block(input_tensor: TensorType, kernel_size, filters, stage, block) -> TensorType:
        """The identity block is the block that has no conv layer at shortcut.

        Args:
            input_tensor: input tensor.
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        Returns:
            output tensor for the block.
        """

        name_base = f"conv{stage}_block{block}_"
        x = layers.conv(input_tensor, filters_out=filters[0], kernel_size=1,
                        add_bias=False, name=name_base + '1_conv')
        x = layers.norm(x, axis=-1, epsilon=1.001e-5, name=name_base + '1_bn')
        x = layers.relu(x, name=name_base + '1_relu')

        x = layers.conv(x, filters_out=filters[1], kernel_size=kernel_size, add_bias=False,
                        name=name_base + '2_conv')
        x = layers.norm(x, axis=-1, epsilon=1.001e-5, name=name_base + '2_bn')
        x = layers.relu(x, name=name_base + '2_relu')

        x = layers.conv(x, filters_out=filters[2], kernel_size=1, add_bias=False,
                        name=name_base + '3_conv')
        x = layers.norm(x, axis=-1, epsilon=1.001e-5, name=name_base + '3_bn')

        x = x + input_tensor
        x = layers.relu(x, name=name_base + '3_relu')

        return x
