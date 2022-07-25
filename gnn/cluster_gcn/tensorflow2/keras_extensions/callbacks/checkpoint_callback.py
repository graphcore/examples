# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import logging

import tensorflow as tf


class CheckpointCallback(tf.keras.callbacks.Callback):

    def __init__(self,
                 checkpoint_dir,
                 executions_per_ckpt):
        self.logger = logging.getLogger("checkpoint_callback")
        self.checkpoint_dir = checkpoint_dir
        self.executions_per_ckpt = executions_per_ckpt

        self.total_number_of_executions = 0

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(self, batch, logs=None):
        self.total_number_of_executions += 1
        if (self.executions_per_ckpt > 0 and
                self.total_number_of_executions % self.executions_per_ckpt == 0):
            self.save_checkpoint(f"execution_{self.total_number_of_executions}")

    def on_train_end(self, logs=None):
        self.save_checkpoint("final")

    def save_checkpoint(self, identifier):
        checkpoint_full_file_path = self.checkpoint_dir.joinpath(f"{identifier}.h5")
        self.logger.info(f"\nSaving checkpoint to: {checkpoint_full_file_path}")
        self.model.save_weights(checkpoint_full_file_path)
