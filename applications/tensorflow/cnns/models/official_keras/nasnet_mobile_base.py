# Copyright 2019 Graphcore Ltd.
"""NASNetMobile model."""

from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from models import tf_layers as layers

ImageNetBlockType = Tuple[int, int, int, int]
CifarBlockType = Tuple[int, int, int]
TensorType = Union[tf.Tensor, np.ndarray, tf.placeholder]


class NASNetMobile(object):
    """Base class for NASNetMobile architecture."""

    def __init__(self,
                 num_classes: int, image_width: Optional[int] = 224, image_height: Optional[int] = 224,
                 image_channels: Optional[int] = 3, weights: Optional[str] = 'imagenet',
                 penultimate_filters: int = 1056, num_blocks: int = 4, stem_block_filters: int = 32,
                 skip_reduction: bool = False, filter_multiplier: int = 2) -> None:
        """Initialize NASNetMobile model.

        Args:

            num_classes: Number of target classes.
            image_width: Input image width.
            image_height: Input image height.
            image_channels: Input image number of channels.
            weights: Initialize with imagenet weights.
            penultimate_filters: Number of filters in the penultimate layer.
            NASNet models use the notation `NASNet (N @ P)`, where:
                -   N is the number of blocks
                -   P is the number of penultimate filters
            num_blocks: Number of repeated blocks of the NASNet model.
                NASNet models use the notation `NASNet (N @ P)`, where:
                    -   N is the number of blocks
                    -   P is the number of penultimate filters
            stem_block_filters: Number of filters in the initial stem block
            skip_reduction: Whether to skip the reduction step at the tail
                end of the network.
            filter_multiplier: Controls the width of the network.
                - If `filter_multiplier` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `filter_multiplier` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `filter_multiplier` = 1, default number of filters from the
                     paper are used at each layer.

        """

        self.num_classes = num_classes
        # Network is set up for NHWC
        self.input_shape = [image_height, image_width, image_channels]
        self.weights = weights
        self.penultimate_filters = penultimate_filters
        self.num_blocks = num_blocks
        self.stem_block_filters = stem_block_filters
        self.skip_reduction = skip_reduction
        self.filter_multiplier = filter_multiplier

    def build_model(self, img_input: TensorType) -> TensorType:
        """Build graph using img_input as input.

        Args:
            img_input: 4D Image input tensor of shape (batch, height, width, channels)

        Returns:
            `Tensor` holding output probabilities per class, shape (batch, num_classes)
        """

        filters = self.penultimate_filters // 24

        x = layers.conv(img_input, filters_out=self.stem_block_filters, kernel_size=3,
                        stride=2,
                        padding='valid',
                        add_bias=False,
                        name='stem_conv1')

        x = layers.norm(x, axis=-1, momentum=0.9997, epsilon=1e-3, name='stem_bn1')

        p = None
        x, p = self.reduction_a_cell(x, p, filters // (self.filter_multiplier ** 2),
                                     block_id='stem_1')
        x, p = self.reduction_a_cell(x, p, filters // self.filter_multiplier,
                                     block_id='stem_2')

        for i in range(self.num_blocks):
            x, p = self.normal_a_cell(x, p, filters, block_id='%d' % i)

        x, p0 = self.reduction_a_cell(x, p, filters * self.filter_multiplier,
                                      block_id='reduce_%d' % self.num_blocks)

        p = p0 if not self.skip_reduction else p

        for i in range(self.num_blocks):
            x, p = self.normal_a_cell(x, p, filters * self.filter_multiplier,
                                      block_id='%d' % (self.num_blocks + i + 1))

        x, p0 = self.reduction_a_cell(x, p, filters * self.filter_multiplier ** 2,
                                      block_id='reduce_%d' % (2 * self.num_blocks))

        p = p0 if not self.skip_reduction else p

        for i in range(self.num_blocks):
            x, p = self.normal_a_cell(x, p, filters * self.filter_multiplier ** 2,
                                      block_id='%d' % (2 * self.num_blocks + i + 1))

        x = layers.relu(x, 'relu')

        # Classification block
        x = layers.avg_pool(x, kernel_size=7, strides=1, name='avg_pool')
        x = layers.squeeze(x, axis=[1, 2], name='squeeze')
        x = layers.fully_connected(x, self.num_classes, name='predictions')
        x = layers.softmax(x, name='output-prob')

        return x

    @staticmethod
    def correct_pad(inputs: tf.Tensor, kernel_size):
        """Returns a tuple for zero-padding for 2D convolution with downsampling.

        Args:
            inputs: An integer or tuple/list of 2 integers.
            kernel_size: An integer or tuple/list of 2 integers.

        Returns:
            A tuple.
        """

        input_size = inputs.get_shape().as_list()[1:3]

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if input_size[0] is None:
            adjust = (1, 1)
        else:
            adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

        correct = (kernel_size[0] // 2, kernel_size[1] // 2)

        return ((correct[0] - adjust[0], correct[0]),
                (correct[1] - adjust[1], correct[1]))

    @staticmethod
    def separable_conv_block(ip, filters,
                             kernel_size=3,
                             strides=1,
                             block_id=None):
        """Adds 2 blocks of [relu-separable conv-batchnorm].

        Args:
            ip: Input tensor
            filters: Number of output filters per layer
            kernel_size: Kernel size of separable convolutions
            strides: Strided convolution for downsampling
            block_id: String block_id

        Returns:
            A tf.Tensor
        """
        channel_dim = -1

        with tf.name_scope('separable_conv_block_%s' % block_id):
            x = layers.relu(ip)
            if strides == (2, 2):
                x = layers.zero_padding(x,
                                        padding=NASNetMobile.correct_pad(x, (kernel_size, kernel_size)),
                                        name='separable_conv_1_pad_%s' % block_id)
                conv_pad = 'valid'
            else:
                conv_pad = 'same'
            x = layers.separable_conv(x, filters_out=filters, kernel_size=kernel_size,
                                      stride=strides,
                                      name='separable_conv_1_%s' % block_id,
                                      padding=conv_pad, add_bias=False)
            x = layers.norm(x,
                            axis=channel_dim,
                            momentum=0.9997,
                            epsilon=1e-3,
                            name='separable_conv_1_bn_%s' % block_id)
            x = layers.relu(x)
            x = layers.separable_conv(x, filters_out=filters, kernel_size=kernel_size,
                                      name='separable_conv_2_%s' % block_id,
                                      padding='same',
                                      add_bias=False)
            x = layers.norm(x,
                            axis=channel_dim,
                            momentum=0.9997,
                            epsilon=1e-3,
                            name='separable_conv_2_bn_%s' % block_id)
        return x

    @staticmethod
    def adjust_block(p, ip, filters, block_id=None):
        """Adjusts the input `previous path` to match the shape of the `input`.

        Used in situations where the output number of filters needs to be changed.

        Args:
            p: Input tensor which needs to be modified
            ip: Input tensor whose shape needs to be matched
            filters: Number of output filters to be matched
            block_id: String block_id

        Returns:
            Adjusted tf tensor.
        """
        channel_dim = -1
        img_dim = -2

        ip_shape = ip.get_shape().as_list()

        if p is not None:
            p_shape = p.get_shape().as_list()
        else:
            p_shape = ip_shape

        with tf.name_scope('adjust_block'):
            if p is None:
                p = ip

            elif p_shape[img_dim] != ip_shape[img_dim]:
                with tf.name_scope('adjust_reduction_block_%s' % block_id):
                    p = layers.relu(p,
                                    name='adjust_relu_1_%s' % block_id)
                    p1 = layers.avg_pool(p,
                                         1,
                                         strides=2,
                                         padding='valid',
                                         name='adjust_avg_pool_1_%s' % block_id)
                    p1 = layers.conv(p1,
                                     filters_out=filters // 2, kernel_size=(1, 1),
                                     padding='same',
                                     add_bias=False, name='adjust_conv_1_%s' % block_id)

                    p2 = layers.zero_padding(p, padding=((0, 1), (0, 1)))
                    p2 = layers.crop(p2, cropping=((1, 0), (1, 0)))
                    p2 = layers.avg_pool(p2,
                                         1,
                                         strides=2,
                                         padding='valid',
                                         name='adjust_avg_pool_2_%s' % block_id)
                    p2 = layers.conv(p2,
                                     filters_out=filters // 2, kernel_size=(1, 1),
                                     padding='same',
                                     add_bias=False,
                                     name='adjust_conv_2_%s' % block_id)

                    p = layers.concat([p1, p2], axis=channel_dim)
                    p = layers.norm(p,
                                    axis=channel_dim,
                                    momentum=0.9997,
                                    epsilon=1e-3,
                                    name='adjust_bn_%s' % block_id)

            elif p_shape[channel_dim] != filters:
                with tf.name_scope('adjust_projection_block_%s' % block_id):
                    p = layers.relu(p)
                    p = layers.conv(p,
                                    filters_out=filters,
                                    kernel_size=(1, 1),
                                    stride=1,
                                    padding='same',
                                    name='adjust_conv_projection_%s' % block_id,
                                    add_bias=False)
                    p = layers.norm(p,
                                    axis=channel_dim,
                                    momentum=0.9997,
                                    epsilon=1e-3,
                                    name='adjust_bn_%s' % block_id)
        return p

    @staticmethod
    def normal_a_cell(ip, p, filters, block_id=None):
        """Adds a Normal cell for NASNet-A (Fig. 4 in the paper).

        Args:
            ip: Input tensor `x`
            p: Input tensor `p`
            filters: Number of output filters
            block_id: String block_id

        Returns:
            A tensorflow tensor
        """
        channel_dim = -1

        with tf.name_scope('normal_A_block_%s' % block_id):
            p = NASNetMobile.adjust_block(p, ip, filters, block_id)

            h = layers.relu(ip)
            h = layers.conv(h,
                            filters_out=filters, kernel_size=(1, 1),
                            stride=1,
                            padding='same',
                            name='normal_conv_1_%s' % block_id,
                            add_bias=False)
            h = layers.norm(h,
                            axis=channel_dim,
                            momentum=0.9997,
                            epsilon=1e-3,
                            name='normal_bn_1_%s' % block_id)

            with tf.name_scope('block_1'):
                x1_1 = NASNetMobile.separable_conv_block(
                    h, filters,
                    kernel_size=5,
                    block_id='normal_left1_%s' % block_id)

                x1_2 = NASNetMobile.separable_conv_block(
                    p, filters,
                    block_id='normal_right1_%s' % block_id)
                x1 = x1_1 + x1_2

            with tf.name_scope('block_2'):
                x2_1 = NASNetMobile.separable_conv_block(
                    p, filters, 5,
                    block_id='normal_left2_%s' % block_id)
                x2_2 = NASNetMobile.separable_conv_block(
                    p, filters, 3,
                    block_id='normal_right2_%s' % block_id)
                x2 = x2_1 + x2_2

            with tf.name_scope('block_3'):
                x3 = layers.avg_pool(h,
                                     3, strides=1,
                                     padding='same',
                                     name='normal_left3_%s' % block_id)
                x3 = x3 + p

            with tf.name_scope('block_4'):
                x4_1 = layers.avg_pool(p,
                                       3, strides=1,
                                       padding='same',
                                       name='normal_left4_%s' % block_id)
                x4_2 = layers.avg_pool(p,
                                       3, strides=1,
                                       padding='same',
                                       name='normal_right4_%s' % block_id)
                x4 = x4_1 + x4_2

            with tf.name_scope('block_5'):
                x5 = NASNetMobile.separable_conv_block(h, filters,
                                                       block_id='normal_left5_%s' % block_id)
                x5 = x5 + h

            x = layers.concat([p, x1, x2, x3, x4, x5],
                              axis=channel_dim,
                              name='normal_concat_%s' % block_id)

        return x, ip

    @staticmethod
    def reduction_a_cell(ip, p, filters, block_id=None):
        """Adds a Reduction cell for NASNet-A (Fig. 4 in the paper).

         Args:
             ip: Input tensor `x`
             p: Input tensor `p`
             filters: Number of output filters
             block_id: String block_id

         Returns:
             A tf tensor
         """
        channel_dim = -1

        with tf.name_scope('reduction_A_block_%s' % block_id):
            p = NASNetMobile.adjust_block(p, ip, filters, block_id)

            h = layers.relu(ip)
            h = layers.conv(h,
                            filters_out=filters, kernel_size=(1, 1),
                            stride=1,
                            padding='same',
                            name='reduction_conv_1_%s' % block_id,
                            add_bias=False)
            h = layers.norm(h,
                            axis=channel_dim,
                            momentum=0.9997,
                            epsilon=1e-3,
                            name='reduction_bn_1_%s' % block_id)

            h3 = layers.zero_padding(h,
                                     padding=NASNetMobile.correct_pad(h, (3, 3)),
                                     name='reduction_pad_1_%s' % block_id)

            with tf.name_scope('block_1'):
                x1_1 = NASNetMobile.separable_conv_block(
                    h, filters=filters, kernel_size=5,
                    strides=2,
                    block_id='reduction_left1_%s' % block_id)
                x1_2 = NASNetMobile.separable_conv_block(
                    p, filters=filters, kernel_size=7,
                    strides=2,
                    block_id='reduction_right1_%s' % block_id)
                x1 = x1_1 + x1_2

            with tf.name_scope('block_2'):
                x2_1 = layers.max_pool(h3, 3,
                                       strides=2,
                                       padding='valid',
                                       name='reduction_left2_%s' % block_id)
                x2_2 = NASNetMobile.separable_conv_block(
                    p, filters=filters, kernel_size=7,
                    strides=2,
                    block_id='reduction_right2_%s' % block_id)
                x2 = x2_1 + x2_2

            with tf.name_scope('block_3'):
                x3_1 = layers.avg_pool(h3,
                                       3,
                                       strides=2,
                                       padding='valid',
                                       name='reduction_left3_%s' % block_id)
                x3_2 = NASNetMobile.separable_conv_block(
                    p, filters, 5,
                    strides=2,
                    block_id='reduction_right3_%s' % block_id)
                x3 = x3_1 + x3_2

            with tf.name_scope('block_4'):
                x4 = layers.avg_pool(x1, 3,
                                     strides=1,
                                     padding='same',
                                     name='reduction_left4_%s' % block_id)
                x4 += x2

            with tf.name_scope('block_5'):
                x5_1 = NASNetMobile.separable_conv_block(
                    x1, filters, 3,
                    block_id='reduction_left4_%s' % block_id)
                x5_2 = layers.max_pool(h3,
                                       3,
                                       strides=2,
                                       padding='valid',
                                       name='reduction_right5_%s' % block_id)
                x5 = x5_1 + x5_2

            x = layers.concat(
                [x2, x3, x4, x5],
                axis=channel_dim,
                name='reduction_concat_%s' % block_id)
            return x, ip


def __call__(self, img_input: TensorType) -> TensorType:
    """Build graph using img_input as input.
    """
    return self.build_model(img_input)
