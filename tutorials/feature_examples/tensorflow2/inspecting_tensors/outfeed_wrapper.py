# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from tensorflow.python.ipu import ipu_outfeed_queue


class MaybeOutfeedQueue:
    """A wrapper for an IPUOutfeedQueue.

    This class allows key-value pairs to be
    selectively added to a dictionary that can then be enqueued.
    If a filter is supplied then one of the filter elements must be
    contained within the key.
    Not tested with replication (behaviour unknown).
    Will not work with non-pipelined Sequential models (known issue).
    """

    def __init__(self, outfeed_mode=None, filters=None):
        """Construct a MaybeOutfeedQueue.

        Args:
            outfeed_mode: The outfeed_mode for the wrapped IPUOutfeedQueue.
            filters: Optional list of strings. If not None then one of these strings
                must be contained within the key for the key,value pair to be added to
                the dictionary that will be enqueued.
        """
        self._queue = ipu_outfeed_queue.IPUOutfeedQueue(outfeed_mode=outfeed_mode)
        self._vals = {}
        self.enqueued = False
        if filters is not None:
            self._filters = []
            self._filters.extend(filters)
        else:
            self._filters = None

    def maybe_outfeed(self, key, value):
        """Potentially add the key,value pair to the internal dictionary.

        If no filters were supplied or if one of the filter strings is
        contained within the key (assumed to be a string) then
        add the key,value pair to the dictionary that will be enqueued.
        """
        if self._filters is not None:
            if any(f in key for f in self._filters):
                self._vals[key] = value
        else:
            self._vals[key] = value

    def maybe_enqueue(self):
        """Potentially enqueue the internal dictionary on the wrapped IPUOutfeedQueue.

        If the dictionary of key,value pairs contains at least one item then
        enqueue the dictionary on the wrapped IPUOutfeedQueue and return the
        enqueue op. Otherwise return None.
        """
        if len(self._vals) > 0:
            self.enqueued = True
            return self._queue.enqueue(self._vals)
        else:
            return None

    def maybe_dequeue(self):
        """Potentially return the dequeue op for the wrapped IPUOutfeedQueue.

        If the wrapped IPUOutfeedQueue has been enqueued, return the
        dequeue op. Otherwise return None.
        """
        if self._queue.enqueued:
            return self._queue.dequeue()
        else:
            return None

    def dequeue(self):
        """Return the dequeue op for the wrapped IPUOutfeedQueue."""
        return self._queue.dequeue()
