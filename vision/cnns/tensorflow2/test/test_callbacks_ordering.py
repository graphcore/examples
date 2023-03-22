# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import tensorflow as tf
import unittest
import threading
import time
import random


class CallbackOrderingTest(unittest.TestCase):
    def get_dataset(self, dataset_size, batch_size):
        x_train = np.ones((dataset_size, 28, 28))
        y_train = np.ones(dataset_size)
        ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        ds = ds.batch(batch_size=batch_size, drop_remainder=True)
        return ds

    def get_model(self):
        initializer = tf.keras.initializers.Ones()
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28)),
                tf.keras.layers.Dense(1, kernel_initializer=initializer, bias_initializer=initializer),
            ]
        )
        model.compile(
            optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
        )
        return model

    def create_callbacks(self, num_callbacks, callback_logs):
        callbacks_to_test = []
        lock = threading.Lock()
        shared_begin_counter = CallbackTestingCounters(lock)
        shared_end_counter = CallbackTestingCounters(lock)
        for i in range(num_callbacks):
            callbacks_to_test.append(CustomTestCallback(shared_begin_counter, shared_end_counter, i, callback_logs))
        return callbacks_to_test

    def get_callback_logs_from_model_training(self, dataset_size, batch_size, num_epochs, num_callbacks, callback_logs):
        model = self.get_model()
        model.fit(
            self.get_dataset(dataset_size, batch_size),
            epochs=num_epochs,
            callbacks=self.create_callbacks(num_callbacks, callback_logs),
        )
        return callback_logs

    def get_expected_callback_logs(self, num_epochs, num_batches, num_callbacks):
        expected_callback_logs = []
        total_num_batches = num_epochs * num_batches
        for batch_index in range(total_num_batches):
            for phase in ["begin", "end"]:
                for callback_index in range(num_callbacks):
                    if phase == "begin":
                        begin_counter_value = num_callbacks * batch_index + (callback_index + 1)
                        expected_callback_logs.append((phase, begin_counter_value, callback_index))
                    else:
                        end_counter_value = num_callbacks * batch_index + (callback_index + 1)
                        expected_callback_logs.append((phase, end_counter_value, callback_index))
        return expected_callback_logs

    def test_callback_ordering(self):
        dataset_size = 600
        batch_size = 50
        num_epochs = 10
        num_callbacks = 10
        num_batches = dataset_size // batch_size
        num_logs = num_epochs * num_batches * num_callbacks * 2
        callback_logs = []

        # callback log entries are in form (phase, begin/end_counter_value, callback_index)
        callback_logs = self.get_callback_logs_from_model_training(
            dataset_size, batch_size, num_epochs, num_callbacks, callback_logs
        )
        self.assertEqual(num_logs, len(callback_logs))

        expected_callback_logs = self.get_expected_callback_logs(num_epochs, num_batches, num_callbacks)
        self.assertEqual(len(callback_logs), len(expected_callback_logs))

        for log, expected_log in zip(callback_logs, expected_callback_logs):
            self.assertEqual(log, expected_log)


class CustomTestCallback(tf.keras.callbacks.Callback):
    def __init__(self, shared_begin_counter, shared_end_counter, callback_index, callback_logs):
        self.callback_index = callback_index
        self.shared_begin_counter = shared_begin_counter
        self.shared_end_counter = shared_end_counter
        self.callback_logs = callback_logs
        self.max_sleep = 0.005  # 5 ms

    def on_batch_begin(self, batch, logs=None):
        time.sleep(random.random() * self.max_sleep)
        self.shared_begin_counter.acquire_lock()
        self.shared_begin_counter.increment_counter()
        self.callback_logs.append(("begin", self.shared_begin_counter.get_counter_value(), self.callback_index))
        self.shared_begin_counter.release_lock()

    def on_batch_end(self, batch, logs=None):
        time.sleep(random.random() * self.max_sleep)
        self.shared_end_counter.acquire_lock()
        self.shared_end_counter.increment_counter()
        self.callback_logs.append(("end", self.shared_end_counter.get_counter_value(), self.callback_index))
        self.shared_end_counter.release_lock()


class CallbackTestingCounters:
    def __init__(self, lock):
        self.counter = 0
        self.lock = lock

    def acquire_lock(self):
        self.lock.acquire()

    def increment_counter(self):
        self.counter += 1

    def get_counter_value(self):
        return self.counter

    def release_lock(self):
        self.lock.release()
