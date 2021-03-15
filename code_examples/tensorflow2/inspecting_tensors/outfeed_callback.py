# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
from tensorflow.keras.callbacks import Callback


class OutfeedCallback(Callback):
    """ Keras callback that dequeues an IPUOutfeedQueue or MaybeOutfeedQueue.

    At the end of each epoch it will call dequeue if data has been
    enqueued on the outfeed queue
    and print the shape and statistics for the tensors contained in the
    dictionary that is returned by the dequeue op.
    """
    def __init__(self, outfeed_queue, name="OutfeedCallback"):
        """ Create an OutfeedCallback.

        It is assumed that the supplied IPUOutfeedQueue or MaybeOutfeedQueue
        will return a dictionary of tensors when the dequeue op is called.
        Args:
            outfeed_queue: An IPUOutfeedQueue or MaybeOutfeedQueue.
            name: Optional name for the callback.
        """
        self._outfeed_queue = outfeed_queue
        self._name = name

    def on_epoch_end(self, epoch, logs=None):
        """ Called at the end of an epoch.

        This function should only be called during TRAIN mode.

        Calls dequeue if data has been enqueued on the outfeed queue.
        Prints the name of the callback.
        Prints the shape and statistics for the tensors contained in the
        dictionary that is returned by the dequeue op.
        If no data has been enqueued then this is reported.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
                validation epoch if validation is performed. Validation result keys
                are prefixed with `val_`.
        """
        print("\n" + self._name)
        if self._outfeed_queue.enqueued:
            data = self._outfeed_queue.dequeue()
            for k, v in data.items():
                print(f"key: {k} shape: {v.shape}")
            self._print_vals(data, epoch + 1)
        else:
            print("No data enqueued")

    def _print_vals(self, vals, epoch):
        data = []
        name_length = np.max([len(name) for name in vals.keys()]) + 5
        for index, (val_name, val) in enumerate(vals.items()):
            data_item = [index]
            data_item.append(val_name)
            data_item.append(f'{np.mean(val):<4.6f}')  # means
            data_item.append(f'{np.std(val):<4.6f}')  # stds
            data_item.append(f'{np.min(val):<4.6f}')  # min extreme
            data_item.append(f'{np.max(val):<4.6f}')  # max extreme
            data_item.append(f'{np.isnan(val).any()}')  # nans?
            data_item.append(f'{np.isinf(val).any()}')  # infs?
            data.append(data_item)

        print(f"Epoch {epoch} - Summary Stats")
        print(f'{"Index":<5} {"Name":<{name_length}} {"Mean":<12} {"Std":<12} {"Minimum":<12} {"Maximum":<12} {"NaNs":<7} {"infs":<7}')
        for index, name, avg, std, dmin, dmax, nans, infs in data:
            print(f"{index:<5} {name:<{name_length}} {avg:<12} {std:<12} {dmin:<12} {dmax:<12} {nans:<7} {infs:<7}")
