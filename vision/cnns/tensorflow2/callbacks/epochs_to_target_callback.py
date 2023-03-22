# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf


class EpochsToTargetCallback(tf.keras.callbacks.Callback):
    def __init__(self, target_field, target_value) -> None:
        super().__init__()
        self.target_field = target_field
        self.target_value = target_value
        self.reached_target = False

    def on_test_batch_end(self, _, logs=None):
        assert self.target_field in logs
        assert "epoch" in logs

        if logs[self.target_field] >= self.target_value and not self.reached_target:
            logs["epoch_to_target"] = logs["epoch"]
            self.reached_target = True
