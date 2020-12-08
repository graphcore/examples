# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
# All contributions by François Chollet:
# Copyright (c) 2015 - 2019, François Chollet.
# All rights reserved.
#
# All contributions by Google:
# Copyright (c) 2015 - 2019, Google, Inc.
# All rights reserved.
#
# All contributions by Microsoft:
# Copyright (c) 2017 - 2019, Microsoft, Inc.
# All rights reserved.
#
# All other contributions:
# Copyright (c) 2015 - 2019, the respective contributors.
# All rights reserved.
#
# Each contributor holds copyright over their respective contributions.
# The project versioning (Git) records all such contribution source information
#
# See https://github.com/keras-team/keras/blob/1a3ee8441933fc007be6b2beb47af67998d50737/examples/cifar10_cnn.py
from functools import partial
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Activation, MaxPooling2D, Flatten


def get_architecture():
    return [Conv2D(32, (3, 3), padding='same'),
            Activation('relu'),
            Conv2D(32, (3, 3)),
            Activation('relu'),
            Conv2D(32, (3, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Conv2D(64, (3, 3), padding='same'),
            Activation('relu'),
            Conv2D(64, (3, 3)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Flatten(),
            Dense(512),
            Activation('relu'),
            Dropout(0.5),
            Dense(10)
            ]


def get_model(logits_only=False):
    model_architecture = get_architecture()
    if not logits_only:
        model_architecture.append(Activation('softmax'))
    return tf.keras.Sequential(model_architecture)


def get_staged_model(is_training, model_shard_position):
    def stage(layers, features, labels):
        for layer in layers:
            if isinstance(layer, Dropout):
                features = layer(features, training=is_training)
            else:
                features = layer(features)
        return features, labels

    model_architecture = get_architecture()
    return [partial(stage, model_architecture[:model_shard_position]),
            partial(stage, model_architecture[model_shard_position:])]
