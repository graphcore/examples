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
            self.queue_buffer = {}

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


def wrap_stateless_metric_in_enqueuer(metric_name, metric_fn, items):
    """
    Wraps a given stateless metric function in functionality for it
    to enqueue the result of its `call` method on an outfeed queue.
    :param metric_name: A name for the metric
    :param metric_fn: A stateless metric method that is callable.
        For example, one that takes y_pred and y_true as arguments
        and returns a value.
    :param items: A list of strings that represent the names of
        the items that are enqueued.
    :return: A class that is able to add the value returned by
        the provided classes `__call__` method onto an outfeed queue.
    """
    if not callable(metric_fn):
        raise NameError(
            f"Provided metric function {metric_fn} is"
            " not callable and so is not suitable for"
            " wrapping in an outfeed enqueuer."
        )

    class OutfeedEnqueuer:
        def __init__(self, *args, **kwargs):
            super(OutfeedEnqueuer, self).__init__(*args, **kwargs)
            self.name = metric_name
            self.outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
            self.items = items
            self.queue_buffer = {}

        def __call__(self, *args, **kwargs):
            """
            Gets the value returned by the provided method_fn __call__
            and adds it to an outfeed queue. It is able to handle a class
            that returns multiple items by constructing a dictionary of the
            values in queue_buffer and only adding to the outfeed once the
            expected number of values have all been added to queue_buffer.
            This is useful if the model contains multiple heads.
            """
            # Get the value
            value = metric_fn(*args, **kwargs)
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

    return OutfeedEnqueuer()


def wrap_stateful_metric_in_enqueuer(metric_class, items):
    """
    Wraps a given class in functionality for it to enqueue
    the result of its `call` method on an outfeed queue. The
    class should be of type `Metric` and can support stateful
    metrics, which have an internal state.
    :param metric_class: Class which will be wrapped such that
        the value returned by its `_fn` method is enqueued to
        an outfeed queue. This class must have an implemented
        `update_state` method.
    :param items: A list of strings that represent the names of
        the items that are enqueued.
    :return: A class that is able to add the value returned by
        the provided classes `call` method onto an outfeed queue.
    """

    call_method = getattr(metric_class, "update_state", None)
    if not callable(call_method):
        raise NameError(
            f"Class {metric_class} does not have an"
            " implemented update_state method and so is not"
            " suitable for wrapping in an outfeed enqueuer."
        )

    class OutfeedEnqueuer(metric_class):
        def __init__(self, *args, **kwargs):
            super(OutfeedEnqueuer, self).__init__(*args, **kwargs)
            self.outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
            self.items = items
            self.queue_buffer = {}

        def update_state(self, y_true, y_pred, sample_weight=None):
            """
            Gets the value returned by the parent classes _fn method
            and adds it to an outfeed queue. It is able to handle a class
            that returns multiple items by constructing a dictionary of the
            values in queue_buffer and only adding to the outfeed once the
            expected number of values have all been added to queue_buffer.
            This is useful if the model contains multiple heads.
            """
            # This implementation doesn't handle using a sample_weight
            if sample_weight is not None:
                raise ValueError(
                    "When using a metric enqueuer, the metric" " should not be calculated with a sample" " weight."
                )
            # Get the value
            value = self._fn(y_true, y_pred, **self._fn_kwargs)
            # Reduce value
            value = tf.reduce_mean(tf.cast(value, self.dtype))
            # Find the corresponding name for that value from self.items
            value_key = self.items[len(self.queue_buffer)]
            # Add the key value pair to a dictionary
            self.queue_buffer[value_key] = value
            # Once all expected values have been added to the queue_buffer
            # dictionary, add to the outfeed queue and clear queue_buffer.
            if len(self.queue_buffer) >= len(self.items):
                self.outfeed_queue.enqueue(self.queue_buffer)
                self.queue_buffer = dict()
            return super().update_state(y_true, y_pred, sample_weight)

    return OutfeedEnqueuer
