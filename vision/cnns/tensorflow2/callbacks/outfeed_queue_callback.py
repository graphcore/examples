# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import numpy as np


class OutFeedQueueCallback(tf.keras.callbacks.Callback):
    def __init__(self, queue, name):
        self._queue = queue
        self.name = name

    def on_train_batch_end(self, _, logs=None):
        self.dequeue_data(_, logs)

    def on_test_batch_end(self, _, logs=None):
        self.dequeue_data(_, logs)

    def dequeue_data(self, _, logs=None):
        if self._queue.enqueued:
            value = self._queue.dequeue()

            if isinstance(value, tf.Tensor):
                value = np.mean(value.numpy())

            logs[self.name] = float(value)
