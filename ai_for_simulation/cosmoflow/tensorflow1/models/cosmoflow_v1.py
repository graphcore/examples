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
