# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Type
import tensorflow as tf
from tensorflow.python import ipu


def wrap_metric_in_enqueuer(
    obj_class: Type[tf.keras.metrics.Metric], outfeed_queue: ipu.ipu_outfeed_queue.IPUOutfeedQueue
) -> Type[tf.keras.metrics.Metric]:
    class OutfeedEnqueuer(obj_class):
        def __init__(self, *args, **kwargs):
            super(OutfeedEnqueuer, self).__init__(*args, **kwargs)
            self.outfeed_queue = outfeed_queue

        def update_state(self, y_true, y_pred, sample_weight=None):
            # perform reduction per micro batch here
            self.outfeed_queue.enqueue(tf.reduce_mean(tf.cast(self._fn(y_true, y_pred, **self._fn_kwargs), self.dtype)))
            return super().update_state(y_true, y_pred, sample_weight)

    return OutfeedEnqueuer


def wrap_metric_in_allreduce_enqueuer(
    obj_class: Type[tf.keras.metrics.Metric], outfeed_queue: ipu.ipu_outfeed_queue.IPUOutfeedQueue, num_replicas: int
) -> Type[tf.keras.metrics.Metric]:
    class AllreduceOutfeedEnqueuer(obj_class):
        def __init__(self, *args, **kwargs):
            super(AllreduceOutfeedEnqueuer, self).__init__(*args, **kwargs)
            self.outfeed_queue = outfeed_queue

        def update_state(self, y_true, y_pred, sample_weight=None):
            # reduction per micro batch
            batch_reduced = tf.reduce_mean(tf.cast(self._fn(y_true, y_pred, **self._fn_kwargs), self.dtype))
            value_to_enqueue = ipu.cross_replica_ops.cross_replica_sum(batch_reduced) / num_replicas
            self.outfeed_queue.enqueue(value_to_enqueue)
            return super().update_state(y_true, y_pred, sample_weight)

    return AllreduceOutfeedEnqueuer
