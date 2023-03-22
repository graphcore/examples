# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import unittest
from pathlib import Path
import sys
import tensorflow as tf

sys.path.append(str(Path(__file__).absolute().parent.parent))
from callbacks.checkpoint_callback import CheckpointCallback
from tensorflow.python import ipu
import os
import shutil
from datetime import datetime
import glob


class TestCheckpointCallbacks(unittest.TestCase):
    def test_saved_validation_list(self):

        root_path = "/tmp"
        now = datetime.now()
        checkpoint_dir = os.path.join(root_path, "checkpoints_test1_" + now.strftime("%d_%m_%Y_%H:%M:%S.%f")[:-3])

        strategy = ipu.ipu_strategy.IPUStrategy()

        num_steps = 2
        num_epochs = 4

        ds = tf.data.Dataset.from_tensor_slices(([1.0, 1.0], [1.0, 2.0]))
        ds = ds.batch(1, drop_remainder=True)
        ds = ds.repeat()

        cfg = ipu.config.IPUConfig()
        cfg.auto_select_ipus = 1
        cfg.device_connection.enable_remote_buffers = True
        cfg.configure_ipu_system()

        callback = CheckpointCallback(ckpt_period=num_steps, ckpt_phase=0, checkpoint_dir=checkpoint_dir)

        weight_value = 160.0

        with strategy.scope():
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
            input_layer = tf.keras.Input(shape=1)
            kernel_initializer = tf.keras.initializers.Constant(weight_value)
            x = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=kernel_initializer)(input_layer)
            model = tf.keras.Model(input_layer, x)
            model.build(input_shape=(1, 1))
            model.compile(optimizer=optimizer, loss=tf.keras.losses.MSE, steps_per_execution=num_steps)
            model.set_gradient_accumulation_options(gradient_accumulation_steps_per_replica=1)
            model.fit(ds, steps_per_epoch=num_steps, epochs=num_epochs, callbacks=callback)

        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.h5"))
        assert len(checkpoint_files) == num_epochs
        with strategy.scope():
            # recompile the model for the validation
            model.compile(metrics=[tf.keras.metrics.MSE], steps_per_execution=num_steps)
            previous_accuracy = 1e10
            for files in sorted(checkpoint_files):
                model.load_weights(files)
                metrics = model.evaluate(ds, steps=num_steps)
                accuracy = metrics[1]
                assert accuracy < previous_accuracy
                previous_accuracy = accuracy
        shutil.rmtree(checkpoint_dir)
