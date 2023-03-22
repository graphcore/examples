# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import logging

import tensorflow as tf

from data_utils.batch_config import BatchConfig


class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        universal_run_name: str,
        checkpoint_dir: str,
        ckpt_every_n_steps_per_execution: int,
        batch_config: BatchConfig,
    ):
        self.logger = logging.getLogger("checkpoint_callback")
        self.universal_run_name = universal_run_name
        self.checkpoint_dir = checkpoint_dir
        self.batch_config = batch_config
        self.ckpt_every_n_steps_per_execution = ckpt_every_n_steps_per_execution

        self.total_number_of_micro_batches = 0

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(self, batch, logs=None):

        self.total_number_of_micro_batches += self.batch_config.steps_per_execution * self.batch_config.num_replicas

        if (
            self.total_number_of_micro_batches
            % (
                self.ckpt_every_n_steps_per_execution
                * self.batch_config.steps_per_execution
                * self.batch_config.num_replicas
            )
            == 0
            or self.total_number_of_micro_batches >= self.batch_config.total_num_micro_batches
        ):
            num_global_batches = (
                self.total_number_of_micro_batches // self.batch_config.num_micro_batches_per_weight_update
            )
            checkpoint_full_file_path = self.checkpoint_dir.joinpath(self.universal_run_name).joinpath(
                f"{num_global_batches}.ckpt"
            )
            self.logger.info(f"\nSaving checkpoint to: {checkpoint_full_file_path}")
            self.model.save_weights(checkpoint_full_file_path)
