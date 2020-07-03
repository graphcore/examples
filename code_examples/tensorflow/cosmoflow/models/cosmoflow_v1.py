"""Model specification for CosmoFlow

This module contains the v1 implementation of the benchmark model.
It is deprecated now and being replaced with the updated, more configurable
architecture currently defined in cosmoflow.py.
"""

import tensorflow as tf
import tensorflow.keras.layers as layers

from .layers import scale_1p2


def build_model(input_shape, target_size, dropout=0):
    """Construct the CosmoFlow 3D CNN model"""

    conv_args = dict(kernel_size=2, padding='valid')

    model = tf.keras.models.Sequential([

        layers.Conv3D(16, input_shape=input_shape, **conv_args),
        layers.LeakyReLU(),
        layers.MaxPool3D(pool_size=2),

        layers.Conv3D(16, **conv_args),
        layers.LeakyReLU(),
        layers.MaxPool3D(pool_size=2),

        layers.Conv3D(16, **conv_args),
        layers.LeakyReLU(),
        layers.MaxPool3D(pool_size=2),

        layers.Conv3D(16, **conv_args),
        layers.LeakyReLU(),
        layers.MaxPool3D(pool_size=2),

        layers.Conv3D(16, **conv_args),
        layers.LeakyReLU(),
        layers.MaxPool3D(pool_size=2),

        layers.Flatten(),
        layers.Dropout(dropout),

        layers.Dense(128),
        layers.LeakyReLU(),
        layers.Dropout(dropout),

        layers.Dense(64),
        layers.LeakyReLU(),
        layers.Dropout(dropout),

        layers.Dense(target_size, activation='tanh'),
        layers.Lambda(scale_1p2)
    ])

    return model
