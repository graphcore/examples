# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Any, Mapping

import tensorflow as tf
from tensorflow.python.ipu.ipu_outfeed_queue import IPUOutfeedQueue
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class LearningRateEnqueuer(LearningRateSchedule):
    def __init__(self, scheduler: LearningRateSchedule, queue: IPUOutfeedQueue):
        super().__init__()
        self._lr_schedule = scheduler
        self._queue = queue

    def __call__(self, step):
        with tf.name_scope("learning_rate_scheduler"):
            new_lr = self._lr_schedule(step)
            self._queue.enqueue({"learning_rate": new_lr})
            return new_lr

    def get_config(self) -> Mapping[str, Any]:
        return self._lr_schedule.get_config()


class FP32StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, schedule: LearningRateSchedule):
        super().__init__()
        self.step = tf.Variable(0, trainable=False, dtype=tf.float32, name="fp32_step")
        self.schedule = schedule

    def __call__(self, _):
        lr = self.schedule(self.step)
        self.step.assign_add(1)
        return lr

    def get_config(self) -> Mapping[str, Any]:
        return {"schedule": self._lr_schedule.get_config()}
