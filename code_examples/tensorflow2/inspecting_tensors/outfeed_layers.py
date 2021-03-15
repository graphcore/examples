# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

""" Keras layers that enqueue inputs on an IPUOutfeedQueue or MaybeOutfeedQueue.
"""

from tensorflow import keras


class Outfeed(keras.layers.Layer):
    """ Keras layer that puts the inputs into a dictionary and enqueues it on an
    IPUOutfeedQueue.

    The outfeed will contain a dictionary that has one entry, where the key is
    the name of the layer and the values are the inputs to it.

    Args:
        outfeed_queue: An IPUOutfeedQueue that is only used by this layer.
    """
    def __init__(self, outfeed_queue, **kwargs):
        super(Outfeed, self).__init__(**kwargs)
        self._outfeed_queue = outfeed_queue

    def call(self, inputs):
        """ Enqueue a dictionary with a single entry on the outfeed queue.

        The key is the name of the layer and the values are the inputs.

        Args:
            inputs: Input tensor (of any rank).

        Returns:
            The inputs.
        """
        self._outfeed_queue.enqueue({self.name: inputs})
        return inputs


class MaybeOutfeed(keras.layers.Layer):
    """ Keras layer that uses a MaybeOutfeedQueue to selectively
    add the inputs to a dict and optionally enqueue this dict.

    The outfeed queue will contain a dictionary where the keys are the names of
    the layers and the values are the inputs to those layers.

    If the MaybeOutfeedQueue was constructed with a list of filters then one of
    the filter elements must be contained within the name of this layer for the
    inputs to be added to the dict.

    When using pipelining, use one MaybeOutfeedQueue per PipelineStage.

    Args:
        maybe_outfeed_queue: A MaybeOutfeedQueue that can be used with more than
            one MaybeOutfeed layers.
        final_outfeed: If True then the dict in the MaybeOutfeedQueue will
            be enqueued. Set this to True for the final MaybeOutfeed layer
            within a PipelineStage or Model/Sequential and False for preceding
            layers that share the maybe_outfeed_queue.

    """
    def __init__(self, maybe_outfeed_queue, final_outfeed=True, **kwargs):
        super(MaybeOutfeed, self).__init__(**kwargs)
        self._outfeed_queue = maybe_outfeed_queue
        self._final_outfeed = final_outfeed

    def call(self, inputs):
        """ Potentially adds the inputs to the outfeed queue and enqueues the
        contents of the queue (depending on other parameters).

        Args:
            inputs: Input tensor (of any rank).

        Returns:
            The inputs.
        """
        self._outfeed_queue.maybe_outfeed(self.name, inputs)
        if self._final_outfeed:
            self._outfeed_queue.maybe_enqueue()

        return inputs
