# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow.python.ipu import distributed
from tensorflow.python.distribute import reduce_util


class AllReduceMetricsCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs):
        self.all_reduce_metrics(batch, logs)

    def on_test_batch_end(self, batch, logs):
        self.all_reduce_metrics(batch, logs)

    def all_reduce_metrics(self, _, logs=None):
        for metric in logs:
            logs[metric] = distributed.allreduce(
                tf.convert_to_tensor(logs[metric], dtype=tf.float32), reduce_util.ReduceOp.MEAN
            ).numpy()
