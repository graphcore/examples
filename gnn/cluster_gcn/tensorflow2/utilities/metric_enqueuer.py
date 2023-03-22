# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow.python import ipu


def wrap_loss_in_enqueuer(loss_class, items):
    """
    Wraps a given class in functionality for it to enqueue
    the result of its `call` method on an outfeed queue.
    :param metric_class: Class which will be wrapped such that
        the value returned by its `call` method is enqueued to
        an outfeed queue. This class must have an implemented
        `call` method.
    :param items: A list of strings that represent the names of
        the items that are enqueued.
    :return: A class that is able to add the value returned by
        the provided classes `call` method onto an outfeed queue.
    """

    call_method = getattr(loss_class, "call", None)
    if not callable(call_method):
        raise NameError(
            f"Class {loss_class} does not have an"
            " implemented call method and so is not"
            " suitable for wrapping in an outfeed"
            " enqueuer."
        )

    class OutfeedEnqueuer(loss_class):
        def __init__(self, *args, **kwargs):
            super(OutfeedEnqueuer, self).__init__(*args, **kwargs)
            self.outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
            self.items = items
            self.queue_buffer = dict()

        def call(self, *args, **kwargs):
            """
            Gets the value returned by the parent classes call method
            and adds it to an outfeed queue. It is able to handle a class
            that returns multiple items by constructing a dictionary of the
            values in queue_buffer and only adding to the outfeed once the
            expected number of values have all been added to queue_buffer.
            This is useful if the model contains multiple heads.
            """
            # Get the value
            value = super().call(*args, **kwargs)
            # Find the corresponding name for that value from self.items
            value_key = self.items[len(self.queue_buffer)]
            # Add the key value pair to a dictionary
            self.queue_buffer[value_key] = value
            # Once all expected values have been added to the queue_buffer
            # dictionary, add to the outfeed queue and clear queue_buffer.
            if len(self.queue_buffer) >= len(self.items):
                self.outfeed_queue.enqueue(self.queue_buffer)
                self.queue_buffer = dict()
            return value

    return OutfeedEnqueuer
