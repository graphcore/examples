# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# Copyright holder unknown (author: François Chollet 2015)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Graphcore Ltd.

import tensorflow.keras as keras
from tensorflow.python import ipu


def model_fn(input_shape, num_classes):
    # Input layer - "entry point" / "source vertex".
    input_layer = keras.Input(shape=input_shape)

    # Add layers to the graph.
    x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(input_layer)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_classes, activation="softmax")(x)

    return input_layer, x


def pipeline_model_fn(input_shape, num_classes):
    # Input layer - "entry point" / "source vertex".
    input_layer = keras.Input(shape=input_shape)

    # Add graph nodes to the first pipeline stage.
    with keras.ipu.PipelineStage(0):
        x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(input_layer)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)

    # Add graph nodes to the second pipeline stage.
    with keras.ipu.PipelineStage(1):
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(num_classes, activation="softmax")(x)

    return input_layer, x
