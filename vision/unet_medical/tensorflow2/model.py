# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Input
from tensorflow.python.ipu import keras as ipu_keras
from model_utils import crop, double_conv_layer, up_conv


def encoder(x, filters, crop_sizes, args):
    skip_connections = []
    for i, filter in enumerate(filters):
        conv = double_conv_layer(filter, x, args.dtype, f"encoder_block_{i}")
        if i > 3:
            conv = ipu_keras.layers.Dropout(
                args.drop_rate, name=f"encoder_block_{i}_IPU_dropout")(conv)
        else:
            # Crop the activation to concatenate later
            cropped_conv = crop(conv, crop_sizes[i], args.nb_ipus_per_replica)
            skip_connections.append(cropped_conv)
            x = MaxPooling2D(pool_size=(
                2, 2), name=f"encoder_block_{i}_maxpooling")(conv)

    return conv, skip_connections


def decoder(x, filters, skip_connections, dtype):
    for i, filter in enumerate(filters[-2::-1]):
        conv = up_conv(filter, x, f"decoder_block_{i}")
        merge = tf.concat([conv, skip_connections.pop()], axis=-1)
        x = double_conv_layer(filter, merge, dtype, f"decoder_block_{i}")
    return x


def model_fn(args):
    inputs = Input((572, 572, 1))
    filters = [64, 128, 256, 512, 1024]
    crop_sizes = [392, 200, 104, 56]
    encoder_result, skip_connections = encoder(inputs, filters, crop_sizes, args)
    decoder_result = decoder(encoder_result, filters, skip_connections, args.dtype)
    conv = Conv2D(args.nb_classes, 1, activation=None,
                  kernel_initializer='he_normal')(decoder_result)
    return inputs, conv
