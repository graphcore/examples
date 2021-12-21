# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))
import normalization


def CifarResNet8(weights=None,
                 input_shape=(32, 32, 3),
                 classes=10):
    return CifarResNet(n=1, input_shape=input_shape, classes=classes)


def CifarResNet20(weights=None,
                  input_shape=(32, 32, 3),
                  classes=10):
    return CifarResNet(n=3, input_shape=input_shape, classes=classes)


def CifarResNet32(weights=None,
                  input_shape=(32, 32, 3),
                  classes=10):
    return CifarResNet(n=5, input_shape=input_shape, classes=classes)


def CifarResNet44(weights=None,
                  input_shape=(32, 32, 3),
                  classes=10):
    return CifarResNet(n=7, input_shape=input_shape, classes=classes)


def CifarResNet56(weights=None,
                  input_shape=(32, 32, 3),
                  classes=10):
    return CifarResNet(n=9, input_shape=input_shape, classes=classes)


def CifarResNet(n, input_shape=(32, 32, 3), classes=10):
    """Instantiates the CifarResNet size dependent architecture - size determined by n.
    Size dependent part consists of: 3 "stacks", each consisting of n "blocks", which each contain 2 "layers" and a residual connection.
    Size independent part: input convolution and output pooling and dense layers."""

    img_input = tf.keras.Input(shape=input_shape)

    x = layer(img_input, filters=16, name='conv1', downsampling=False)
    x = tf.keras.layers.ReLU(name='conv1_relu')(x)

    x = stack(x, 16, n, downsampling=False, name='conv2')
    x = stack(x, 32, n, downsampling=True, name='conv3')
    x = stack(x, 64, n, downsampling=True, name='conv4')

    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dense(classes, activation=tf.keras.activations.softmax, name='fc_layer')(x)

    model = tf.keras.Model(img_input, x, name='cifar_resnet_model')
    return model


def stack(x, filters, n, downsampling, name=None):
    """A stack consists of n blocks which each contain a residual connection."""

    x = block(x, filters, downsampling, name=f'{name}_block1')
    for i in range(2, n+1):
        x = block(x, filters, downsampling=False, name=f'{name}_block{i}')
    return x


def block(x, filters, downsampling=False, name=None):
    """A block consists of 2 weighted convolutional layers."""

    shortcut = residual_connection(x, filters, f'{name}_shortcut', downsampling)

    x = layer(x, filters, f'{name}_layer1', downsampling)
    x = tf.keras.layers.ReLU(name=f'{name}_layer1_relu')(x)

    x = layer(x, filters, f'{name}_layer2', downsampling=False)

    x = tf.keras.layers.Add(name=f'{name}_add')([shortcut, x])
    x = tf.keras.layers.ReLU(name=f'{name}_post_addition_relu')(x)
    return x


def layer(x, filters, name, downsampling):
    """2D convolution layer with a 3x3 kernel.
    Downsampled using a stride of 2."""

    stride, name_append = (2, '_conv_ds') if downsampling else (1, '_conv')
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=stride, padding='SAME',
                               use_bias=False, name=f'{name}{name_append}',
                               kernel_initializer=tf.keras.initializers.VarianceScaling())(x)

    x = normalization.batch_norm.BatchNormIPU(axis=3, epsilon=1.001e-5, name=f'{name}_bn')(x)
    return x


def residual_connection(x, filters, name, downsampling=False):
    """Added as a shortcut over a block of 2 layers.
    Downsampled using 2D convolution with a 1x1 kernel and stride of 2."""

    if downsampling:
        batches, height, width, channels = x.shape
        assert(channels*2 == filters)
        shortcut = x[:, 0:height:2, 0:width:2, :]
        shortcut = tf.pad(shortcut, paddings=[[0, 0], [0, 0], [0, 0], [0, channels]], name=f'{name}_pad')
        assert(shortcut.shape[1:] == (height//2, width//2, channels*2))
    else:
        shortcut = x
    return shortcut
