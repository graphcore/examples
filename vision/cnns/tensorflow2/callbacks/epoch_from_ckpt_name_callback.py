# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import re


class EpochFromCkptNameCallback(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_list) -> None:
        self.ckpt_list = ckpt_list
        self.idx = 0
        self.pattern = re.compile(".*epoch_(.*)\.h5")

    def on_test_end(self, logs=None):
        self.idx += 1

    def on_test_batch_end(self, batch, logs=None):
        if len(self.ckpt_list):
            ckpt_name = self.ckpt_list[self.idx]
            match = re.match(self.pattern, ckpt_name)
            if match:
                epoch = float(match.group(1))
                logs["epoch"] = epoch
            else:
                raise ValueError(f"{ckpt_name} has unexpected pattern, could not extract epoch")
        else:
            logs["epoch"] = float(self.idx)
