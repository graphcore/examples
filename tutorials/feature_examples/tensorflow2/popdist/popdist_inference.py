# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popdist
import popdist.tensorflow

import numpy as np

import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.ipu.distributed import popdist_strategy

BATCH_SIZE = 32

# Initialize IPU configuration.
config = ipu.config.IPUConfig()
popdist.tensorflow.set_ipu_config(config)
config.configure_ipu_system()

# Create distribution strategy.
strategy = popdist_strategy.PopDistStrategy()

# Get and normalize the testing data.
_, (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
test_x = test_x.astype(np.float32) / 255.0
test_y = test_y.astype(np.int32)

# Create dataset and shard it across the instances.
dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
dataset = dataset.shard(num_shards=popdist.getNumInstances(), index=popdist.getInstanceIndex())

# Perform batching after sharding.
for i, element in enumerate(dataset):
    if i == 0:
        print(element)
dataset = dataset.batch(batch_size=BATCH_SIZE, drop_remainder=True)

# Calculate steps per execution, these methods are equivalent
total_batches = len(test_y) // BATCH_SIZE
steps_per_execution = total_batches // popdist.getNumTotalReplicas()

shard_batches = len(dataset)
steps_per_execution = shard_batches // popdist.getNumLocalReplicas()

if popdist.getInstanceIndex() == 0:
    num_samples = len(test_y)
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
    model = tf.keras.models.load_model("saved_model.h5")

    model.compile(metrics=["accuracy"], steps_per_execution=steps_per_execution)
    # Equally, you could use `model.predict()`
    hist = model.evaluate(dataset, verbose=1)
