# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import math
from random import randint
import unittest
import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from schedules.scheduler_factory import get_lr_scheduler
from schedules.decorators import ShiftWarmup, FadingMaskWarmup, StairCase
from schedules.lr_schedules import CosineLRSchedule, ConstLRSchedule
from callbacks.outfeed_queue_callback import OutFeedQueueCallback


SUPPORTED_LR_SCHEDULES = {
    "cosine": (CosineLRSchedule, {"initial_learning_rate": 0.0001, "epochs_to_total_decay": 1.1}),
    "stepped": (PiecewiseConstantDecay, {"boundaries": [0.1, 0.5, 0.8], "values": [0.00001, 0.0001, 0.0005, 0.00001]}),
    "const": (
        ConstLRSchedule,
        {
            "initial_learning_rate": 0.0001,
        },
    ),
}


class DummyScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self):
        super().__init__()
        self.initial_learning_rate = 0.5

    def __call__(self, step):
        lr = self.initial_learning_rate + step
        return lr


class LearningRateSchedulesTest(unittest.TestCase):
    def test_stepped_lr_schedule_output(self):
        step_boundaries = [2, 5, 10]
        lr_values = [0.1, 0.2, 0.3, 0.4]
        schedule = PiecewiseConstantDecay(step_boundaries, lr_values)

        step_boundaries = step_boundaries.copy()
        step_boundaries.insert(0, -1)
        step_boundaries.append(step_boundaries[-1] * 2)

        for i, lr_value in enumerate(lr_values):
            step = randint(step_boundaries[i] + 1, step_boundaries[i + 1])
            assert schedule(step) == lr_value

    def test_cosine_lr_schedule_output(self):
        tolerance = 1e-6

        steps_to_total_decay = 50
        initial_learning_rate = 2

        schedule = CosineLRSchedule(initial_learning_rate, steps_to_total_decay)

        errors = []
        for step in range(steps_to_total_decay):
            lr = schedule(step).numpy()
            cosine_reference = initial_learning_rate * 0.5 * (1 + math.cos((step * math.pi) / steps_to_total_decay))
            errors += [abs(lr - cosine_reference)]

        assert all(val < tolerance for val in errors)

    def test_getting_supported_lr_schedules(self):
        for scheduler_name, (cls, params) in SUPPORTED_LR_SCHEDULES.items():
            schedule = get_lr_scheduler(
                scheduler_name=scheduler_name, schedule_params=params, global_batch_size=1, weight_updates_per_epoch=1
            )
            assert type(schedule.schedule) == cls

    def test_error_for_unsupported_lr_schedules(self):
        try:
            get_lr_scheduler(
                scheduler_name="hygge", schedule_params={}, global_batch_size=1, weight_updates_per_epoch=1
            )
        except NameError:
            assert True
            return
        assert False

    def test_lr_scheduler_builder_factor(self):
        scheduler = get_lr_scheduler(
            scheduler_name="const",
            schedule_params={"initial_learning_rate": 2},
            global_batch_size=1,
            weight_updates_per_epoch=1,
            factor=10,
        )
        self.assertEqual(scheduler(0), 20)


class ScheduleDecoratorsTest(unittest.TestCase):
    def __init__(self, methodName: str):
        super().__init__(methodName=methodName)
        self.schedule = DummyScheduler()

    @staticmethod
    def run_fn_on_device(fn, args=[]):
        cfg = ipu.config.IPUConfig()
        cfg.device_connection.enable_remote_buffers = True
        cfg.configure_ipu_system()

        strategy = ipu.ipu_strategy.IPUStrategy()
        with strategy.scope():
            return strategy.run(fn, args=args)

    def test_shift_warmup_decorator(self):
        warmup_weight_updates = 9
        scheduler_with_warmup = ShiftWarmup(scheduler=self.schedule, warmup_weight_updates=warmup_weight_updates)

        assert scheduler_with_warmup(0) == self.schedule.initial_learning_rate / 10
        assert scheduler_with_warmup(1) == self.schedule.initial_learning_rate * 2 / 10
        assert scheduler_with_warmup(2) == self.schedule.initial_learning_rate * 3 / 10
        assert scheduler_with_warmup(3) == self.schedule.initial_learning_rate * 4 / 10
        assert scheduler_with_warmup((warmup_weight_updates - 1) / 2) == self.schedule.initial_learning_rate / 2
        assert scheduler_with_warmup(warmup_weight_updates) == self.schedule(0)
        assert scheduler_with_warmup(warmup_weight_updates + 1) == self.schedule(1)

    def test_fading_mask_warmup_decorator(self):
        warmup_weight_updates = 9
        scheduler_with_warmup = FadingMaskWarmup(scheduler=self.schedule, warmup_weight_updates=warmup_weight_updates)

        assert scheduler_with_warmup(0) == self.schedule.initial_learning_rate / 10
        assert (
            scheduler_with_warmup((warmup_weight_updates - 1) / 2) == self.schedule((warmup_weight_updates - 1) / 2) / 2
        )
        assert scheduler_with_warmup(warmup_weight_updates) == self.schedule(warmup_weight_updates)

    def test_staircase_decorator(self):
        weight_updates_per_stair_tread = 2
        schedule_with_staircase = StairCase(self.schedule, weight_updates_per_stair_tread)
        num_steps = 10
        for step in range(num_steps):
            self.assertEqual(
                schedule_with_staircase(step),
                self.schedule(step // weight_updates_per_stair_tread * weight_updates_per_stair_tread),
                msg=f"failed on step {step}",
            )


class LRCallbackTest(unittest.TestCase):
    def enqueue_and_dequeue(self, to_enqueue) -> dict:
        cfg = ipu.config.IPUConfig()
        cfg.device_connection.enable_remote_buffers = True
        cfg.configure_ipu_system()
        outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
        callback = OutFeedQueueCallback(queue=outfeed_queue, name="lr")

        @tf.function(experimental_compile=True)
        def fn():
            outfeed_queue.enqueue(to_enqueue)

        strategy = ipu.ipu_strategy.IPUStrategy()
        with strategy.scope():
            strategy.run(fn)
            logs = {}
            callback.on_train_batch_end(0, logs=logs)

        return logs

    def test_lr_callback_for_single_replica(self):
        to_enqueue = 3.141590118408203
        logs = self.enqueue_and_dequeue(to_enqueue)
        assert logs["lr"] == to_enqueue

    def test_lr_callback_for_many_replicas(self):
        to_enqueue = tf.convert_to_tensor([3.14159, 3.14159, 3.14159])
        logs = self.enqueue_and_dequeue(to_enqueue)
        self.assertAlmostEqual(logs["lr"], to_enqueue[0], places=4)
