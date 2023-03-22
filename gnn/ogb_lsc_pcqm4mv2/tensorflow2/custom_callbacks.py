# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging
import os
import tensorflow as tf
import wandb


class CheckpointCallback(tf.keras.callbacks.ModelCheckpoint):
    def __init__(
        self, use_wandb=False, upload_to_wandb=False, save_checkpoints_locally=False, total_epochs=None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_wandb = use_wandb
        self.upload_to_wandb = upload_to_wandb
        self.save_checkpoints_locally = save_checkpoints_locally
        self.epochs = total_epochs

    def on_train_end(self, epoch, logs=None):
        """Overwrite the on train end method to save the last checkpoint
        and then save the checkpoint to wandb
        """
        filepath = self.filepath.replace("{epoch:05d}", "FINAL")
        self.model.save_weights(filepath, overwrite=True, options=self._options)

        if self.use_wandb and self.upload_to_wandb:
            logging.info(f"Saving model weights from {filepath} to wandb...")
            # Save all model checkpoints with string from above
            if self.save_checkpoints_locally:
                # This allows the saving format to be the same when coming from a tmp dir
                splits = filepath.split("/")
                base_path = os.path.join(*splits[:-1])
            else:
                # If the checkpoint is saved in 'tmp/' no base_path is needed
                base_path = None
            # Final checkpoints uploaded to wandb in root directory of wandb run
            wandb.save(filepath + "*", policy="now", base_path=base_path)
