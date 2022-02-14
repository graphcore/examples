# 'Regression of 3D Sky Map to Cosmological Parameters (CosmoFlow)'
# Copyright (c) 2018, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy).  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Innovation & Partnerships Office at IPO@lbl.gov.
#
# NOTICE.  This Software was developed under funding from the U.S. Department of
# Energy and the U.S. Government consequently retains certain rights. As such,
# the U.S. Government has been granted for itself and others acting on its
# behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
# to reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit other to do so.

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
