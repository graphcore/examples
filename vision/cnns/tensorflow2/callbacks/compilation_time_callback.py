# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import sys
import time

import tensorflow as tf


class CompilationTimeCallback(tf.keras.callbacks.Callback):
    def __init__(self, compile_only: bool = False):
        self.compile_only = compile_only
        self.__current_batch_operations = self.__calculate_compilation_time

    def on_train_begin(self, logs=None):
        self.compilation_start_time = time.time()

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        self.__current_batch_operations(logs)

    def __do_nothing(self, logs):
        pass

    def __calculate_compilation_time(self, logs):
        if logs is not None:
            logs["Compilation Time"] = time.time() - self.compilation_start_time
        self.__current_batch_operations = self.__do_nothing

        if self.compile_only:
            sys.exit(0)
