# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow.python.ipu import horovod as hvd


class AllReduceMetricsCallback(tf.keras.callbacks.Callback):

    def on_train_batch_end(self, batch, logs):
        self.all_reduce_metrics(batch, logs)

    def on_test_batch_end(self, batch, logs):
        self.all_reduce_metrics(batch, logs)

    def all_reduce_metrics(self, _, logs=None):
        for metric in logs:
            logs[metric] = hvd.allreduce(tf.convert_to_tensor(logs[metric], dtype=tf.float32)).numpy()
