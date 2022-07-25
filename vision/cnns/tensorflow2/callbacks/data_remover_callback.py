# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import typing
import logging


class DataRemoverCallback(tf.keras.callbacks.Callback):

    def __init__(self, fields_to_remove: typing.List[str]):
        self.fields_to_remove = fields_to_remove

    def remove_fields(self, logs: typing.Optional[dict]=None):
        if logs:
            for key in self.fields_to_remove:
                logs.pop(key, None)

    def on_train_batch_end(self, batch: int, logs: typing.Optional[dict]=None):
        self.remove_fields(logs)

    def on_test_batch_end(self, batch: int, logs: typing.Optional[dict]=None):
        self.remove_fields(logs)
