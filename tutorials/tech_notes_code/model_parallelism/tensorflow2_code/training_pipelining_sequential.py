# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.ipu import utils
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.python.ipu import config
from tensorflow.python.ipu import ipu_strategy
import numpy as np
import tensorflow as tf

# default data_format is 'channels_last'
dataset = Dataset.from_tensor_slices(
    (
        tf.random.uniform([2, 128, 128, 3], dtype=tf.float32),
        tf.random.uniform([2], maxval=10, dtype=tf.int32),
    )
)
dataset = dataset.batch(batch_size=2, drop_remainder=True)
dataset = dataset.shuffle(1000)
dataset = dataset.cache()
dataset = dataset.repeat()
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)


# Create a pipelined model which is split across four stages.
def my_model():
    return tf.keras.Sequential(
        [
            layers.Conv2D(3, 1),
            layers.Conv2D(3, 1),
            layers.Conv2D(3, 1),
            layers.Flatten(),
            layers.Dense(10),
        ]
    )


cfg = config.IPUConfig()
cfg.allow_recompute = True
cfg.selection_order = config.SelectionOrder.ZIGZAG
cfg.auto_select_ipus = 4
cfg.configure_ipu_system()
utils.move_variable_initialization_to_cpu()


# Define the model under an IPU strategy scope
strategy = ipu_strategy.IPUStrategy()
with strategy.scope():
    model = my_model()

    model.set_pipelining_options(gradient_accumulation_steps_per_replica=8)
    model.set_pipeline_stage_assignment([0, 1, 2, 3, 3])

    model.compile(
        steps_per_execution=128,
        loss="sparse_categorical_crossentropy",
        optimizer=optimizers.SGD(0.01),
    )

    model.fit(dataset, steps_per_epoch=128)
