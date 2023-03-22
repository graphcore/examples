# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import time


class CompilationTimeCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.__current_batch_operations = self.__calculate_compilation_time

    def on_train_begin(self, logs=None):
        self.compilation_start_time = time.time()

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        self.__current_batch_operations(logs)

    def __do_nothing(self, logs):
        pass

    def __calculate_compilation_time(self, logs):
        if logs is not None:
            logs["Compilation Time"] = time.time() - self.compilation_start_time
        self.__current_batch_operations = self.__do_nothing
