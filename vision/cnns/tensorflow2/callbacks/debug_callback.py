# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow.python import ipu


class DebugCallback(tf.keras.callbacks.Callback):
    def __init__(self, queue: ipu.ipu_outfeed_queue.IPUOutfeedQueue, name: str):
        self._queue = queue
        self.name = name

    def on_train_batch_end(self, _, __):
        self.dequeue_data()

    def on_test_batch_end(self, _, __):
        self.dequeue_data()

    def dequeue_data(self):
        if self._queue.enqueued:
            value = self._queue.dequeue()
            print(f"{self.name}: {value}")
