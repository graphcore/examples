# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import math

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class CosineLRSchedule(LearningRateSchedule):
    def __init__(self, initial_learning_rate: float, weight_updates_to_total_decay: int):

        super(CosineLRSchedule, self).__init__()
        self._weight_updates_to_total_decay = float(weight_updates_to_total_decay)
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        step = tf.minimum(step, self._weight_updates_to_total_decay)
        lr = self.initial_learning_rate * 0.5 * (1 + tf.cos((step * math.pi) / self._weight_updates_to_total_decay))
        return lr


class ConstLRSchedule(LearningRateSchedule):
    def __init__(self, initial_learning_rate: float = 1e-3):

        super(ConstLRSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return step * 0 + self.initial_learning_rate
