# Copyright 2020 Graphcore Ltd.
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from models.tf_layers import conv_norm_relu, max_pool, avg_pool, concat, \
    squeeze, fully_connected, softmax

ImageNetBlockType = Tuple[int, int, int, int]
CifarBlockType = Tuple[int, int, int]
TensorType = Union[tf.Tensor, np.ndarray, tf.placeholder]


class InceptionV3(object):
    """Base class for InceptionV3 architecture."""

    def __init__(self,
                 num_classes: int, image_width: Optional[int] = 299, image_height: Optional[int] = 299,
                 image_channels: Optional[int] = 3, weights: Optional[str] = 'imagenet') -> None:
        """Initialize inceptionv3 model.

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

        x = conv_norm_relu(img_input, 32, 3, strides=2, padding='VALID')
        x = conv_norm_relu(x, 32, 3, padding='VALID')
        x = conv_norm_relu(x, 64, 3, )
        x = max_pool(x, 3, strides=2)

        x = conv_norm_relu(x, 80, 1, padding='VALID')
        x = conv_norm_relu(x, 192, 3, padding='VALID')
        x = max_pool(x, 3, strides=2)

        # mixed 0: 35 x 35 x 256
        branch1x1 = conv_norm_relu(x, 64, 1)

        branch5x5 = conv_norm_relu(x, 48, 1)
        branch5x5 = conv_norm_relu(branch5x5, 64, 5)

        branch3x3dbl = conv_norm_relu(x, 64, 1)
        branch3x3dbl = conv_norm_relu(branch3x3dbl, 96, 3)
        branch3x3dbl = conv_norm_relu(branch3x3dbl, 96, 3)

        branch_pool = avg_pool(x, 3, strides=1, padding='SAME')
        branch_pool = conv_norm_relu(branch_pool, 32, 1)
        x = concat(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=-1,
            name='mixed0')

        # mixed 1: 35 x 35 x 288
        branch1x1 = conv_norm_relu(x, 64, 1)

        branch5x5 = conv_norm_relu(x, 48, 1)
        branch5x5 = conv_norm_relu(branch5x5, 64, 5)

        branch3x3dbl = conv_norm_relu(x, 64, 1)
        branch3x3dbl = conv_norm_relu(branch3x3dbl, 96, 3)
        branch3x3dbl = conv_norm_relu(branch3x3dbl, 96, 3)

        branch_pool = avg_pool(x, 3, strides=1, padding='SAME')
        branch_pool = conv_norm_relu(branch_pool, 64, 1)
        x = concat(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=-1,
            name='mixed1')

        # mixed 2: 35 x 35 x 288
        branch1x1 = conv_norm_relu(x, 64, 1)

        branch5x5 = conv_norm_relu(x, 48, 1)
        branch5x5 = conv_norm_relu(branch5x5, 64, 5)

        branch3x3dbl = conv_norm_relu(x, 64, 1)
        branch3x3dbl = conv_norm_relu(branch3x3dbl, 96, 3)
        branch3x3dbl = conv_norm_relu(branch3x3dbl, 96, 3)

        branch_pool = avg_pool(x, 3, strides=1, padding='SAME')
        branch_pool = conv_norm_relu(branch_pool, 64, 1)
        x = concat(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=-1,
            name='mixed2')

        # mixed 3: 17 x 17 x 768
        branch3x3 = conv_norm_relu(x, 384, 3, strides=2, padding='VALID')

        branch3x3dbl = conv_norm_relu(x, 64, 1)
        branch3x3dbl = conv_norm_relu(branch3x3dbl, 96, 3)
        branch3x3dbl = conv_norm_relu(branch3x3dbl, 96, 3, strides=2, padding='VALID')

        branch_pool = max_pool(x, 3, 2)
        x = concat(
            [branch3x3, branch3x3dbl, branch_pool],
            axis=-1,
            name='mixed3')

        # mixed 4: 17 x 17 x 768
        branch1x1 = conv_norm_relu(x, 192, 1)

        branch7x7 = conv_norm_relu(x, 128, 1)
        branch7x7 = conv_norm_relu(branch7x7, 128, 1, 7)
        branch7x7 = conv_norm_relu(branch7x7, 192, 7, 1)

        branch7x7dbl = conv_norm_relu(x, 128, 1, 1)
        branch7x7dbl = conv_norm_relu(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = conv_norm_relu(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = conv_norm_relu(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = conv_norm_relu(branch7x7dbl, 192, 1, 7)

        branch_pool = avg_pool(x, 3, strides=1, padding='SAME')
        branch_pool = conv_norm_relu(branch_pool, 192, 1, 1)
        x = concat(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=-1,
            name='mixed4')

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = conv_norm_relu(x, 192, 1, 1)

            branch7x7 = conv_norm_relu(x, 160, 1, 1)
            branch7x7 = conv_norm_relu(branch7x7, 160, 1, 7)
            branch7x7 = conv_norm_relu(branch7x7, 192, 7, 1)

            branch7x7dbl = conv_norm_relu(x, 160, 1, 1)
            branch7x7dbl = conv_norm_relu(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv_norm_relu(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = conv_norm_relu(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv_norm_relu(branch7x7dbl, 192, 1, 7)

            branch_pool = avg_pool(x, 3, strides=1, padding='SAME')
            branch_pool = conv_norm_relu(branch_pool, 192, 1, 1)
            x = concat(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=-1,
                name='mixed' + str(5 + i))

        # mixed 7: 17 x 17 x 768
        branch1x1 = conv_norm_relu(x, 192, 1, 1)

        branch7x7 = conv_norm_relu(x, 192, 1, 1)
        branch7x7 = conv_norm_relu(branch7x7, 192, 1, 7)
        branch7x7 = conv_norm_relu(branch7x7, 192, 7, 1)

        branch7x7dbl = conv_norm_relu(x, 192, 1, 1)
        branch7x7dbl = conv_norm_relu(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = conv_norm_relu(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = conv_norm_relu(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = conv_norm_relu(branch7x7dbl, 192, 1, 7)

        branch_pool = avg_pool(x, 3, strides=1, padding='SAME')
        branch_pool = conv_norm_relu(branch_pool, 192, 1, 1)
        x = concat(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=-1,
            name='mixed7')

        # mixed 8: 8 x 8 x 1280
        branch3x3 = conv_norm_relu(x, 192, 1)
        branch3x3 = conv_norm_relu(branch3x3, 320, 3,
                                   strides=2, padding='VALID')

        branch7x7x3 = conv_norm_relu(x, 192, 1, 1)
        branch7x7x3 = conv_norm_relu(branch7x7x3, 192, 1, 7)
        branch7x7x3 = conv_norm_relu(branch7x7x3, 192, 7, 1)
        branch7x7x3 = conv_norm_relu(
            branch7x7x3, 192, 3, 3, strides=2, padding='VALID')

        branch_pool = max_pool(x, 3, strides=2)
        x = concat(
            [branch3x3, branch7x7x3, branch_pool],
            axis=-1,
            name='mixed8')

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = conv_norm_relu(x, 320, 1, 1)

            branch3x3 = conv_norm_relu(x, 384, 1, 1)
            branch3x3_1 = conv_norm_relu(branch3x3, 384, 1, 3)
            branch3x3_2 = conv_norm_relu(branch3x3, 384, 3, 1)
            branch3x3 = concat(
                [branch3x3_1, branch3x3_2],
                axis=-1,
                name='mixed9_' + str(i))

            branch3x3dbl = conv_norm_relu(x, 448, 1, 1)
            branch3x3dbl = conv_norm_relu(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = conv_norm_relu(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = conv_norm_relu(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = concat([branch3x3dbl_1, branch3x3dbl_2], axis=-1)

            branch_pool = avg_pool(x, 3, strides=1, padding='SAME')
            branch_pool = conv_norm_relu(branch_pool, 192, 1, 1)
            x = concat(
                [branch1x1, branch3x3, branch3x3dbl, branch_pool],
                axis=-1,
                name='mixed' + str(9 + i))

        # Classification block
        x = avg_pool(x, kernel_size=8, strides=1, name='avg_pool')
        x = squeeze(x, axis=[1, 2], name='squeeze')
        x = fully_connected(x, self.num_classes, name='predictions')
        x = softmax(x, name='output-prob')

        return x

    def __call__(self, img_input: TensorType) -> TensorType:
        """Build graph using img_input as input.
        """
        return self.build_model(img_input)
