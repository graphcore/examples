# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from tensorflow import keras
import time


class ThroughputCallback(keras.callbacks.Callback):
    def __init__(self, samples_per_epoch):
        super().__init__()
        self.time = 0
        self.samples_per_epoch = samples_per_epoch

    def on_epoch_begin(self, epoch, logs=None):
        self.time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        time_per_epoch = time.time() - self.time
        samples_per_sec = self.samples_per_epoch / time_per_epoch
        print(f"Throughput: {samples_per_sec:.2f} samples/sec")


def size_hr(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def print_trainable_variables(model):
    print('Trainable Variables:')
    total_parameters = 0
    total_size = 0
    for variable in model.trainable_variables:
        variable_parameters = 1
        for DIM in variable.shape:
            variable_parameters *= DIM
        variable_size = variable_parameters * variable.dtype.size
        print(f"{variable.name}, {variable.shape}, {variable.dtype} ({size_hr(variable_size)})")
        total_parameters += variable_parameters
        total_size += variable_size
    print(f"Total Parameters: {total_parameters:,}  ({size_hr(total_size)})")
