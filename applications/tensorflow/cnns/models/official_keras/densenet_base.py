# Copyright 2019 Graphcore Ltd.

from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf


from models import tf_layers


ImageNetBlockType = Tuple[int, int, int, int]
CifarBlockType = Tuple[int, int, int]
TensorType = Union[tf.Tensor, np.ndarray, tf.placeholder]

layers = tf_layers

# TODO(lakshmik): Add popart backend


class DenseNet(object):
    """Base class for DenseNet architecture."""

    def __init__(self, blocks: Union[ImageNetBlockType, CifarBlockType],
                 num_classes: int, image_width: int, image_height: int,
                 image_channels: Optional[int] = 3, weights: Optional[str] = 'imagenet') -> None:
        """Initialize densenet model.

        Args:
            blocks: Number of building blocks within each dense block.
            num_classes: Number of target classes.
            image_width: Input image width.
            image_height: Input image height.
            image_channels: Input image number of channels.
            weights: Initialize with imagenet weights.

        """
        self.blocks = blocks
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

        x = layers.conv(img_input, filters_out=64, kernel_size=7, stride=2, add_bias=False, name='conv1/conv')
        x = layers.norm(x, axis=-1, epsilon=1.001e-5, name='conv1/bn')
        x = layers.relu(x, name='conv1/relu')
        x = layers.zero_padding(x, padding=((1, 1), (1, 1)), name='zero_padding_2d_2')
        x = layers.max_pool(x, kernel_size=3, name='pool1')

        for i, block_size in enumerate(self.blocks):
            x = self.dense_block(x, block_size, name=f'conv{i+2}')
            if (i+1) != len(self.blocks):
                x = self.transition_block(x, 0.5, name=f'pool{i+2}')

        x = layers.norm(x, axis=-1, epsilon=1.001e-5)
        x = layers.relu(x, name='relu')
        x = layers.avg_pool(x, kernel_size=7, strides=1, name='avg_pool')
        x = layers.squeeze(x, axis=[1, 2], name='squeeze')
        x = layers.fully_connected(x, self.num_classes, name='fc1000')
        x = layers.softmax(x, name='output-prob')
        return x

    def __call__(self, img_input: TensorType) -> TensorType:
        """Build graph using img_input as input.
        """
        return self.build_model(img_input)

    @staticmethod
    def conv_block(x: Union[tf.Tensor, np.ndarray], growth_rate: int, name: str) -> TensorType:
        """Building block for a dense block.

        Args:
            x: Input tensor of type tf.Tensor if using tf backend, np.ndarray if using popart builder.
            growth_rate: Growth rate at dense layers.
            name: Block label.

        Return:
            Output tensor for the block.
        """

        x1 = layers.norm(x, axis=-1, epsilon=1.001e-5, name=name + '_0_bn')
        x1 = layers.relu(x1, name=name + '_0_relu')
        x1 = layers.conv(x1, filters_out=4 * growth_rate, kernel_size=1, add_bias=False, name=name + '_1_conv')
        x1 = layers.norm(x1, axis=-1, epsilon=1.001e-5, name=name + '_1_bn')
        x1 = layers.relu(x1, name=name + '_1_relu')
        x1 = layers.conv(x1, filters_out=growth_rate, kernel_size=3, add_bias=False, name=name + '_2_conv')
        x = layers.concat([x, x1], axis=-1, name=name + "_concat")
        return x

    def dense_block(self, x: TensorType, blocks: int, name: str) -> TensorType:
        """A dense block.

        Args:
            x: input tensor.
            blocks: the number of building blocks.
            name: block label.

        Returns:
            output tensor for the block.
        """
        for i in range(blocks):
            x = self.conv_block(x, growth_rate=32, name=name + '_block' + str(i + 1))
        return x

    @staticmethod
    def transition_block(x: TensorType, reduction: float, name: str) -> TensorType:
        """A transition block.

        Args:
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.

        Returns:
            output tensor for the block.
        """
        x = layers.norm(x, axis=-1, epsilon=1.001e-5, name=name + "_bn")
        x = layers.relu(x, name=name + "_relu")
        x = layers.conv(x, filters_out=int(x.get_shape().as_list()[-1] * reduction), kernel_size=1, add_bias=False,
                        name=name + "_conv")
        x = layers.avg_pool(x, kernel_size=2, strides=2, name=name)
        return x
