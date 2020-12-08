# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10


class CIFAR10_Data:
    def __init__(self):
        self._load_data()

    def _load_data(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
        self.x_train, self.x_test = x_train / 255.0, x_test / 255.0
        y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
        self.y_train, self.y_test = y_train.astype(np.int32), y_test.astype(np.int32)

    def get_train_dataset(self, batch_size):
        return tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).shuffle(10000).batch(batch_size, drop_remainder=True).repeat()

    def get_test_dataset(self, batch_size):
        return tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(batch_size, drop_remainder=True)

    @staticmethod
    def _make_generator_dataset(x, y, batch_size):
        def generator():
            return zip(x, y)
        types = (x.dtype, y.dtype)
        shapes = (x.shape[1:], y.shape[1:])
        ds = tf.data.Dataset.from_generator(generator, types, shapes)
        ds = ds.prefetch(len(x)).cache().repeat().shuffle(len(x)).batch(batch_size, drop_remainder=True)
        return ds

    def get_train_datagenerator(self, batch_size):
        return CIFAR10_Data._make_generator_dataset(self.x_train, self.y_train, batch_size)

    def get_test_datagenerator(self, batch_size):
        return CIFAR10_Data._make_generator_dataset(self.x_test, self.y_test, batch_size)
