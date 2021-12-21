# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Type
import tensorflow as tf
from tensorflow.python import ipu


def wrap_loss_in_enqueuer(obj_class: Type[tf.keras.losses.Loss],
                          outfeed_queue: ipu.ipu_outfeed_queue.IPUOutfeedQueue
                          ) -> Type[tf.keras.losses.Loss]:

    class OutfeedEnqueuer(obj_class):

        def __init__(self, *args, **kwargs):
            super(OutfeedEnqueuer, self).__init__(*args, **kwargs)
            self.outfeed_queue = outfeed_queue

        def call(self, *args, **kwargs):
            value = super().call(*args, **kwargs)
            self.outfeed_queue.enqueue(value)
            return value

    return OutfeedEnqueuer
