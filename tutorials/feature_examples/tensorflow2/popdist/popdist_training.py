# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popdist
import popdist.tensorflow

import numpy as np

import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.ipu.distributed import popdist_strategy

# Note that this is the per replica batch size. The batch size for each
# training step will be `BATCH_SIZE * popdist.getNumTotalReplicas()`
BATCH_SIZE = 32
LEARNING_RATE = 0.01
NUM_EPOCHS = 100


def initialize_model():
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10),
        ]
    )


# Initialize IPU configuration.
config = ipu.config.IPUConfig()
popdist.tensorflow.set_ipu_config(config)
config.configure_ipu_system()

# Create distribution strategy.
strategy = popdist_strategy.PopDistStrategy()

# Get and normalize the training data.
(train_x, train_y), _ = tf.keras.datasets.cifar10.load_data()
train_x = train_x.astype(np.float32) / 255.0
train_y = train_y.astype(np.int32)

# Create dataset and shard it across the instances.
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.shard(num_shards=popdist.getNumInstances(), index=popdist.getInstanceIndex())

# Perform shuffling and batching after sharding.
dataset = dataset.shuffle(buffer_size=len(dataset))
dataset = dataset.batch(batch_size=BATCH_SIZE, drop_remainder=True)

# Calculate steps per execution, these methods are equivalent
total_batches = len(train_y) // BATCH_SIZE
steps_per_execution = total_batches // popdist.getNumTotalReplicas()

shard_batches = len(dataset)
steps_per_execution = shard_batches // popdist.getNumLocalReplicas()

if popdist.getInstanceIndex() == 0:
    num_samples = len(train_y)
    print(f"Number of samples:\t{num_samples}")

    print(f"Data distribution between {popdist.getNumInstances()} instances:")
    samples_per_shard = num_samples // popdist.getNumInstances()
    print(f"\tSamples per shard:\t{samples_per_shard}")
    batches_per_shard = samples_per_shard // BATCH_SIZE
    print(f"\tBatches per shard:\t{batches_per_shard}")
    samples_discarded = samples_per_shard % BATCH_SIZE
    print(f"\tSamples discarded:\t{samples_discarded}")

    print(f"Data distribution within each instance ({popdist.getNumLocalReplicas()} local replica(s)):")
    batches_per_replica = batches_per_shard // popdist.getNumLocalReplicas()
    print(f"\tBatches per replica:\t{batches_per_replica}")
    batches_discarded = batches_per_shard % popdist.getNumLocalReplicas()
    print(f"\tBatches discarded:\t{batches_discarded}")

    print(f"Steps per execution:\t{steps_per_execution}")

with strategy.scope():
    model = initialize_model()
    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
        steps_per_execution=steps_per_execution,
    )
    history = model.fit(dataset, epochs=NUM_EPOCHS)

    if popdist.getInstanceIndex() == 0:
        model.save(f"saved_model.h5")
