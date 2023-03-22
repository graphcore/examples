# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import tensorflow as tf


class OutFeedQueueCallback(tf.keras.callbacks.Callback):
    def __init__(self, queue):
        self._queue = queue

    def on_train_batch_end(self, _, logs=None):
        self.dequeue_data(_, logs)

    def on_test_batch_end(self, _, logs=None):
        self.dequeue_data(_, logs)

    def dequeue_data(self, _, logs=None):
        if self._queue.enqueued:
            value_dict = self._queue.dequeue()

            for k, v in value_dict.items():
                if isinstance(v, tf.Tensor):
                    v = np.mean(v.numpy())
                logs[k] = v
