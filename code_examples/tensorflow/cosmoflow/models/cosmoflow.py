# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

"""Configurable model specification for CosmoFlow"""

import tensorflow as tf
import tensorflow.keras.layers as layers

from .layers import scale_1p2


def build_model(input_shape, target_size,
                conv_size=16, kernel_size=2, n_conv_layers=5,
                fc1_size=128, fc2_size=64,
                hidden_activation='LeakyReLU',
                pooling_type='MaxPool3D',
                dropout=0):
    """Construct the CosmoFlow 3D CNN model"""

    conv_args = dict(kernel_size=kernel_size, padding='same')
    hidden_activation = getattr(layers, hidden_activation)
    pooling_type = getattr(layers, pooling_type)

    model = tf.keras.models.Sequential()

    # First convolutional layer
    model.add(layers.Conv3D(conv_size, input_shape=input_shape, **conv_args))
    model.add(hidden_activation())
    model.add(pooling_type(pool_size=2))

    # Additional conv layers
    for i in range(1, n_conv_layers):
        # Double conv channels at every layer
        model.add(layers.Conv3D(conv_size*2**i, **conv_args))
        model.add(hidden_activation())
        model.add(pooling_type(pool_size=2))
    model.add(layers.Flatten())

    # Fully-connected layers
    model.add(layers.Dense(fc1_size))
    model.add(hidden_activation())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(fc2_size))
    model.add(hidden_activation())
    model.add(layers.Dropout(dropout))

    # Output layers
    model.add(layers.Dense(target_size, activation='tanh'))
    model.add(layers.Lambda(scale_1p2))

    return model
