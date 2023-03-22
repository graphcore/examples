# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from tensorflow.python.keras.applications import resnet
from tensorflow.keras import backend
import tensorflow as tf
from tensorflow.keras import layers

import logging

logger = logging.getLogger("resnet_models")


def vanilla_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False, use_bias=False, name=None):
    """A residual block with a shortcut of 2.

    Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      use_bias: boolean, enables or disables all biases in conv layers.
      name: string, block label.

    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            use_bias=use_bias,
            name=f"{name}_0_conv",
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
        )(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=f"{name}_0_bn")(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="SAME",
        strides=stride,
        use_bias=use_bias,
        name=f"{name}_1_conv",
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=f"{name}_1_bn")(x)
    x = layers.Activation("relu", name=f"{name}_1_relu")(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="SAME",
        use_bias=use_bias,
        name=f"{name}_2_conv",
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=f"{name}_2_bn")(x)

    x = layers.Add(name=f"{name}_add")([shortcut, x])
    x = layers.Activation("relu", name=f"{name}_out")(x)
    return x


def bottleneck_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False, use_bias=False, name=None):
    """A residual block.

    Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.

    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * filters,
            1,
            strides=stride,
            use_bias=use_bias,
            name=f"{name}_0_conv",
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
        )(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=f"{name}_0_bn")(shortcut)
        bn_gamma_initializer = "ones"
    else:
        shortcut = x
        bn_gamma_initializer = "zeros"

    x = layers.Conv2D(
        filters,
        1,
        1,
        use_bias=use_bias,
        name=f"{name}_1_conv",
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=f"{name}_1_bn")(x)
    x = layers.Activation("relu", name=f"{name}_1_relu")(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        stride,
        padding="SAME",
        use_bias=use_bias,
        name=f"{name}_2_conv",
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=f"{name}_2_bn")(x)
    x = layers.Activation("relu", name=f"{name}_2_relu")(x)

    x = layers.Conv2D(
        4 * filters,
        1,
        1,
        use_bias=use_bias,
        name=f"{name}_3_conv",
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, gamma_initializer=bn_gamma_initializer, epsilon=1.001e-5, name=f"{name}_3_bn"
    )(x)

    x = layers.Add(name=f"{name}_add")([shortcut, x])
    x = layers.Activation("relu", name=f"{name}_out")(x)
    return x


def stack(x, filters, num_blocks, block_fn, use_bias, stride1=2, name=None):
    """A set of stacked residual blocks.

    Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      use_bias: boolean, enables or disables all biases in conv layers.
      num_blocks: integer, blocks in the stacked blocks.
      block_fn: a function of a block to stack.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.

    Returns:
      Output tensor for the stacked blocks.
    """
    x = block_fn(x, filters, conv_shortcut=True, stride=stride1, use_bias=use_bias, name=f"{name}_block1")
    for i in range(2, num_blocks + 1):
        x = block_fn(x, filters, use_bias=use_bias, name=f"{name}_block{i}")
    return x


def ResNet18(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    use_bias=False,
    block_fn=vanilla_block,
    **kwargs,
):
    """Instantiates the ResNet18 architecture."""

    def stack_fn(x):
        x = stack(x, 64, 2, block_fn=block_fn, use_bias=use_bias, stride1=1, name="conv2")
        x = stack(x, 128, 2, block_fn=block_fn, use_bias=use_bias, name="conv3")
        x = stack(x, 256, 2, block_fn=block_fn, use_bias=use_bias, name="conv4")
        return stack(x, 512, 2, block_fn=block_fn, use_bias=use_bias, name="conv5")

    return resnet.ResNet(
        stack_fn,
        False,
        use_bias,
        "resnet18",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        layers=layers,
        **kwargs,
    )


def ResNet34(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    use_bias=False,
    block_fn=vanilla_block,
    **kwargs,
):
    """Instantiates the ResNet34 architecture."""

    def stack_fn(x):
        x = stack(x, 64, 3, block_fn=block_fn, use_bias=use_bias, stride1=1, name="conv2")
        x = stack(x, 128, 4, block_fn=block_fn, use_bias=use_bias, name="conv3")
        x = stack(x, 256, 6, block_fn=block_fn, use_bias=use_bias, name="conv4")
        return stack(x, 512, 3, block_fn=block_fn, use_bias=use_bias, name="conv5")

    return resnet.ResNet(
        stack_fn,
        False,
        use_bias,
        "resnet34",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        layers=layers,
        **kwargs,
    )


def ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    use_bias=False,
    block_fn=bottleneck_block,
    **kwargs,
):
    """Instantiates the ResNet50 architecture."""

    def stack_fn(x):
        x = stack(x, 64, 3, block_fn=block_fn, use_bias=use_bias, stride1=1, name="conv2")
        x = stack(x, 128, 4, block_fn=block_fn, use_bias=use_bias, name="conv3")
        x = stack(x, 256, 6, block_fn=block_fn, use_bias=use_bias, name="conv4")
        return stack(x, 512, 3, block_fn=block_fn, use_bias=use_bias, name="conv5")

    return resnet.ResNet(
        stack_fn,
        False,
        use_bias,
        "custom_resnet50",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        layers=layers,
        **kwargs,
    )
