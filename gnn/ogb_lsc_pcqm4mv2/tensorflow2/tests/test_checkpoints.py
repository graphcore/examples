# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import inspect
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory

import tensorflow as tf
from tensorflow.python import ipu

from custom_callbacks import CheckpointCallback


@dataclass
class model:
    epochs: int = 120
    model: str = "Fake_Model"


@dataclass
class Config:
    model: model = model()


def test_checkpoint_creation():
    with TemporaryDirectory() as temp_dir:

        checkpoint_name = f"{inspect.currentframe().f_code.co_name}" f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_dir = Path(temp_dir).joinpath(checkpoint_name)

        micro_batch_size = 4
        steps_per_execution = 5
        num_replicas = 2
        total_num_micro_batches = 130
        checkpoint_frequency = 50

        ds = tf.data.Dataset.from_tensor_slices(([1.0, 1.0], [1.0, 2.0]))
        ds = ds.repeat()
        ds = ds.batch(micro_batch_size, drop_remainder=True)

        ipu_cfg = ipu.config.IPUConfig()
        ipu_cfg.auto_select_ipus = num_replicas
        ipu_cfg.device_connection.type = ipu.utils.DeviceConnectionType.ON_DEMAND
        ipu_cfg.configure_ipu_system()

        fake_cfg = Config

        logging.info("Running training...")
        logging.info(f"Saving weights to {checkpoint_dir}")
        model_path = os.path.join(checkpoint_dir, "model-{epoch:05d}")

        callback = CheckpointCallback(
            use_wandb=False,
            upload_to_wandb=False,
            save_checkpoints_locally=False,
            filepath=model_path,
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            period=checkpoint_frequency,
            total_epochs=fake_cfg.model.epochs,
        )
        strategy = ipu.ipu_strategy.IPUStrategy()
        with strategy.scope():
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
            input_layer = tf.keras.Input(shape=1)
            x = tf.keras.layers.Dense(1, use_bias=False)(input_layer)
            model = tf.keras.Model(input_layer, x)
            model.build(input_shape=(1, 1))
            model.compile(optimizer=optimizer, loss=tf.keras.losses.MSE, steps_per_execution=steps_per_execution)
            model.fit(ds, steps_per_epoch=total_num_micro_batches, epochs=fake_cfg.model.epochs, callbacks=callback)

        checkpoint_files = glob(str(checkpoint_dir.joinpath("*.index")))

        # 2 checkpoints created at 50 and 100 micro batches
        # 1 checkpoint created at end of training (120 micro batches)
        assert len(checkpoint_files) == 3
        assert any(["FINAL" in file for file in checkpoint_files])
