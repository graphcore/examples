# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import psutil


class CPUMemoryCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        if logs is not None:
            logs["%CPU memory"] = psutil.virtual_memory().percent

    def on_test_batch_end(self, batch, logs=None):
        if logs is not None:
            logs["%CPU memory"] = psutil.virtual_memory().percent
