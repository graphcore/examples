# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
#
# This file has been modified by Graphcore Ltd.

import time
import tensorflow as tf
from tensorflow import keras

from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import config
from tensorflow.python.ipu import ipu_strategy


step_count = 10_000
steps_per_execution = 10


# Create the input data and labels
def create_dataset():
    mnist = keras.datasets.mnist

    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = x_train / 255.0

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(len(x_train)).batch(32, drop_remainder=True)
    train_ds = train_ds.map(lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.int32)))
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds.repeat()


# Create a simple fully-connected network model using the standard Keras Sequential API
def create_model():
    m = keras.Sequential(
        [
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return m


@tf.function(experimental_compile=True)
def training_loop(iterator, outfeed_queue, model, optimizer, steps_per_execution):
    for _ in tf.range(steps_per_execution):
        # Get the data for the step.
        features, labels = next(iterator)

        # Perform the training step.
        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            prediction_loss = keras.losses.sparse_categorical_crossentropy(labels, predictions)
            loss = tf.reduce_mean(prediction_loss)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Store the loss in the outfeed queue.
        outfeed_queue.enqueue(loss)


# Configure the IPU system
cfg = config.IPUConfig()
cfg.auto_select_ipus = 1
cfg.configure_ipu_system()

# Create an IPU distribution strategy
strategy = ipu_strategy.IPUStrategy()

with strategy.scope():
    # An optimizer for updating the trainable variables.
    opt = keras.optimizers.SGD(0.01)

    # Create an instance of the model.
    model = create_model()

    # Create an iterator for the dataset.
    iterator = iter(create_dataset())

    # Create an IPUOutfeedQueue to collect results from each on device step.
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    start_time = time.time()
    # Train the model
    for _ in range(0, step_count, steps_per_execution):

        # Run `steps_per_execution` at a time.
        strategy.run(
            training_loop,
            args=[iterator, outfeed_queue, model, opt, steps_per_execution],
        )
    result = outfeed_queue.dequeue()
    print("Time taken using infeed/outfeed queues:", time.time() - start_time, "seconds")
