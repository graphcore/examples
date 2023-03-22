# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Any, Mapping

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class ShiftWarmup(LearningRateSchedule):
    def __init__(self, scheduler: LearningRateSchedule, warmup_weight_updates: int):
        """
        Linearly approaches 'initial_learning_rate' over 'warmup_weight_updates'.
        The wrapped scheduler takes over when 'steps > warmup_weight_updates',
        starting from step 0.

        Args:
            scheduler: an object of class 'LearningRateSchedule' to be wrapped
            warmup_weight_updates: number of warming up steps (step=weight update)
        """
        super(ShiftWarmup, self).__init__()

        self._lr_schedule = scheduler
        self._warmup_steps = float(warmup_weight_updates)

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)

        lr = tf.cond(
            step >= self._warmup_steps,
            lambda: self._lr_schedule(step - self._warmup_steps),
            lambda: self._lr_schedule(0) * (step + 1) / (self._warmup_steps + 1),
        )

        return lr

    def get_config(self) -> Mapping[str, Any]:
        config = self._lr_schedule.get_config()
        config.update({"warmup_steps": self._warmup_weight_updates, "warmup_lr": self._warmup_lr})
        return config


class FadingMaskWarmup(LearningRateSchedule):
    def __init__(self, scheduler: LearningRateSchedule, warmup_weight_updates: int):
        """
        Applies a mask over the wrapped scheduler's output.
        The mask gradually fades out over 'warmup_weight_updates' steps,
        fading out entirely when 'step > warmup_weight_updates'.

        Args:
            scheduler: an object of class 'LearningRateSchedule' to be wrapped
            warmup_steps: number of warming up steps
        """
        super(FadingMaskWarmup, self).__init__()

        self._lr_schedule = scheduler
        self._warmup_steps = float(warmup_weight_updates)

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)

        lr = tf.cond(
            step >= self._warmup_steps,
            lambda: self._lr_schedule(step),
            lambda: self._lr_schedule(step) * (step + 1) / (self._warmup_steps + 1),
        )

        return lr

    def get_config(self) -> Mapping[str, Any]:
        config = self._lr_schedule.get_config()
        config.update({"warmup_steps": self._warmup_weight_updates})
        return config


class FP32StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, schedule: LearningRateSchedule):
        super(FP32StepLearningRateSchedule, self).__init__()
        self.step = tf.Variable(0, trainable=False, dtype=tf.float32, name="fp32_step")
        self.schedule = schedule

    def __call__(self, _):
        lr = self.schedule(self.step)
        self.step.assign_add(1)
        return lr

    def get_config(self) -> Mapping[str, Any]:
        return {"schedule": self._lr_schedule.get_config()}


class StairCase(LearningRateSchedule):
    def __init__(self, lr_schedule, weight_updates_per_stair_tread: int = 1):
        super(StairCase, self).__init__()
        self.lr_schedule = lr_schedule
        self.weight_updates_per_stair_tread = weight_updates_per_stair_tread

    def __call__(self, step):
        staircased_step = step // self.weight_updates_per_stair_tread * self.weight_updates_per_stair_tread
        return self.lr_schedule(staircased_step)

    def get_config(self) -> Mapping[str, Any]:
        return {
            "schedule": self._lr_schedule.get_config(),
            "weight_updates_per_stair_tread": self.weight_updates_per_stair_tread,
        }
