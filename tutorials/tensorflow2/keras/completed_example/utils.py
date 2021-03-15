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

import argparse

import numpy as np
import tensorflow.keras as keras


def load_data():
    # Load the MNIST dataset from keras.datasets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize the images.
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # When dealing with images, we usually want an explicit channel dimension, even when it is 1.
    # Each sample thus has a shape of (28, 28, 1).
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Finally, convert class assignments to a binary class matrix.
    # Each row can be seen as a rank-1 "one-hot" tensor.
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def parse_params():
    parser = argparse.ArgumentParser(description='Keras MNIST example')
    parser.add_argument("--pipelining", action="store_true",
                        help="Enable IPU pipelining")
    parser.add_argument("--use-ipu", action="store_true",
                        help="Use IPU-specific Keras model classes")
    parser.add_argument("--gradient-accumulation-count", type=int, default=8,
                        help="The number of times each pipeline stage will be executed")
    args = parser.parse_args()
    if (args.pipelining or args.gradient_accumulation_count) and not args.use_ipu:
        print("Note: `--pipelining` and `--gradient-accumulation-count` are IPU"
              " specific flags. See `--use-ipu`")
    if (args.gradient_accumulation_count and not args.pipelining):
        print("Note: `--gradient-accumulation-count` controls the number of"
              " mini-batches that flow through the pipeline before a weight"
              " update. Supply the `--pipelining` flag to use it.")
    return args
