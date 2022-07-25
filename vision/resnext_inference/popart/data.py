# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import numpy as np


class DataSet:
    def __init__(self, tensors, batch_size, device_iterations,
                 loader, dtype=np.float16):
        self.tensors = tensors
        self.dtype = dtype
        self.loader = loader
        self.num_examples = len(loader) * batch_size * device_iterations
        self.batch_size = batch_size
        self.device_iterations = min(device_iterations,
                                     self.num_examples //
                                     self.batch_size)
        self.inputs_per_step = self.batch_size * self.device_iterations
        self.steps_per_epoch = self.num_examples // self.inputs_per_step

        # Determine the shape of the batch based on batch size
        # and replication factor
        self.batch_shape = [batch_size]

        if self.device_iterations > 1:
            self.batch_shape = [self.device_iterations] + self.batch_shape

        # This needs to be done here as the DataLoader will fork the workers.
        # Fork does not work well once the program has started
        self.loader_iterator = self.loader.__iter__()

    def __iter__(self):
        self.loader_iterator.reset()
        return self

    def __len__(self):
        return self.steps_per_epoch

    def __next__(self):

        # Get the next image/label
        items = next(self.loader_iterator)

        # Reshape the input
        items = map(lambda item: item.reshape(self.batch_shape + list(item.shape[1:])), items)

        return dict(zip(self.tensors, items))
