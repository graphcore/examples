# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.keras.backend import epsilon


class SmoothedCategoricalCrossentropy(tf.keras.losses.Loss):

    def __init__(self, num_classes, label_smoothing, *args, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        one_d_labels = tf.cast(tf.transpose(y_true)[0], tf.int32)
        micro_batch_size = len(one_d_labels)
        epsilon_ = tf.cast(epsilon(), y_pred.dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon_, 1. - epsilon_)
        y_true = tf.one_hot(one_d_labels, self.num_classes, dtype=y_pred.dtype, axis=-1)
        y_true = y_true * (1. - self.label_smoothing) + (self.label_smoothing / self.num_classes)

        if y_true.shape != y_pred.shape:

            raise ValueError(f'Label shape is {y_true.shape} while prediction shape is {y_pred.shape}')

        return -math_ops.reduce_sum(y_true * math_ops.log(y_pred)) / micro_batch_size
