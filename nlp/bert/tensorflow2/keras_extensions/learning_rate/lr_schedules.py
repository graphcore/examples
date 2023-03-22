# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class LearningRateWarmupAndDecay(LearningRateSchedule):
    def __init__(self, warmup_frac: float, total_steps: int, max_learning_rate: float):
        super().__init__()
        self.num_warmup_steps = int(warmup_frac * total_steps)
        self.total_steps = total_steps
        self.max_learning_rate = max_learning_rate

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)

        def ramp_up(step):
            return (step / self.num_warmup_steps) * self.max_learning_rate

        def ramp_down(step):
            return self.max_learning_rate * ((self.total_steps - step - 1) / (self.total_steps - self.num_warmup_steps))

        lr = tf.cond(step <= self.num_warmup_steps, lambda: ramp_up(step), lambda: ramp_down(step))

        # In order to avoid a LR of exactly zero, a minimal learning rate of 1e-7 is imposed
        lr = tf.reduce_max([1e-7, lr])
        return lr
