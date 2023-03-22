# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Type
import tensorflow as tf
from tensorflow.python import ipu


def wrap_loss_in_enqueuer(
    obj_class: Type[tf.keras.losses.Loss], loss_outfeed_queue: ipu.ipu_outfeed_queue.IPUOutfeedQueue
) -> Type[tf.keras.losses.Loss]:
    class OutfeedEnqueuer(obj_class):
        def __init__(self, *args, **kwargs):
            super(OutfeedEnqueuer, self).__init__(*args, **kwargs)
            self.loss_outfeed_queue = loss_outfeed_queue

        def call(self, *args, **kwargs):
            value = super().call(*args, **kwargs)
            self.loss_outfeed_queue.enqueue(value)
            return value

    return OutfeedEnqueuer


def wrap_loss_in_allreduce_enqueuer(
    obj_class: Type[tf.keras.losses.Loss],
    outfeed_queue: ipu.ipu_outfeed_queue.IPUOutfeedQueue,
    num_replicas: int,
) -> Type[tf.keras.losses.Loss]:
    class AllReduceEnqueuer(obj_class):
        def __init__(self, *args, **kwargs):
            super(AllReduceEnqueuer, self).__init__(*args, **kwargs)
            self.outfeed_queue = outfeed_queue

        def call(self, *args, **kwargs):
            value = super().call(*args, **kwargs)
            value_to_enqueue = ipu.cross_replica_ops.cross_replica_sum(value) / num_replicas
            self.outfeed_queue.enqueue(value_to_enqueue)

            return value

    return AllReduceEnqueuer


def wrap_loss_in_pred_enqueuer(
    obj_class: Type[tf.keras.losses.Loss], pred_outfeed_queue: ipu.ipu_outfeed_queue.IPUOutfeedQueue
) -> Type[tf.keras.losses.Loss]:
    class PredOutfeedEnqueuer(obj_class):
        def __init__(self, *args, **kwargs):
            super(PredOutfeedEnqueuer, self).__init__(*args, **kwargs)
            self.pred_outfeed_queue = pred_outfeed_queue

        def call(self, y_true, y_pred, **kwargs):
            value = super().call(y_true, y_pred, **kwargs)
            self.pred_outfeed_queue.enqueue(y_pred)
            return value

    return PredOutfeedEnqueuer


def wrap_loss_in_label_enqueuer(
    obj_class: Type[tf.keras.losses.Loss], label_outfeed_queue: ipu.ipu_outfeed_queue.IPUOutfeedQueue
) -> Type[tf.keras.losses.Loss]:
    class LabelOutfeedEnqueuer(obj_class):
        def __init__(self, *args, **kwargs):
            super(LabelOutfeedEnqueuer, self).__init__(*args, **kwargs)
            self.label_outfeed_queue = label_outfeed_queue

        def call(self, y_true, y_pred, **kwargs):
            value = super().call(y_true, y_pred, **kwargs)
            self.label_outfeed_queue.enqueue(y_true)
            return value

    return LabelOutfeedEnqueuer
