# Copyright (c) 2021 Graphcore Ltd. All rights reserved.


def wrap_loss_in_enqueuer(loss_class, outfeed_queue, items):

    class OutfeedEnqueuer(loss_class):

        def __init__(self, *args, **kwargs):
            super(OutfeedEnqueuer, self).__init__(*args, **kwargs)
            self.outfeed_queue = outfeed_queue
            self.items = items
            self.queue_buffer = {}

        def call(self, *args, **kwargs):
            value = super().call(*args, **kwargs)
            self.queue_buffer[self.items[len(self.queue_buffer)]] = value
            if len(self.queue_buffer) >= len(self.items):
                self.outfeed_queue.enqueue(self.queue_buffer)
                self.queue_buffer = {}
            return value

    return OutfeedEnqueuer
