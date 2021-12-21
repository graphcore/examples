# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np
from bert_optimizer import LinearOptimizerFactory
from tests.unit.optimizer.optimizer_factory_test import MockIteration, TestConfig
from bert import Iteration


def linear_schedule(start, end, interval, low, high):
    update_steps = np.arange(start, end, interval)
    updates = np.linspace(low, high, len(update_steps))
    return dict(zip(update_steps, updates))


def run_linear_optimizer_factory_case(config, iteration, epoch_truth=None, step_truth=None, option_name="defaultLearningRate"):
    """Runs a single case of the linear optimizer factory tests. Simulates running through
    every step of every epoch required by the config, and updates the optimizer factory as
    defined by the schedule. Then checks the optimizer parameters to ensure they are correct."""
    if epoch_truth is None:
        epoch_truth = {}

    if step_truth is None:
        step_truth = {}

    factory = LinearOptimizerFactory(config, iteration)
    for iteration.epoch in range(iteration.epochs):
        for _ in range(iteration.steps_per_epoch):
            if factory.should_update(iteration):
                factory.update(iteration)

            if iteration.count in step_truth:
                print(iteration.count)
                lr = factory.option_values["defaultLearningRate"]
                assert lr == step_truth[iteration.count]
            iteration.count += 1


@pytest.mark.parametrize(
    "warmup, steps_per_decay, learning_rate",
    [(0, 16, 1), (10, 2, 3e-2), (30, 50, 8e-4)])
def test_linear_optimizer_factory(warmup, steps_per_decay, learning_rate):
    iteration = MockIteration(20, 100)
    config = TestConfig(
        learning_rate_function="Linear",
        lr_warmup_steps=warmup,
        lr_steps_per_decay_update=steps_per_decay,
        learning_rate=learning_rate
    )

    def generate_step_truth():
        if warmup > 0:
            schedule = linear_schedule(0, warmup, 1, 1e-7, learning_rate)
        else:
            schedule = {}
        schedule.update(linear_schedule(warmup, iteration.total_steps, steps_per_decay, learning_rate, 1e-7))
        return schedule

    step_truth = generate_step_truth()
    run_linear_optimizer_factory_case(config, iteration, step_truth=step_truth)


@pytest.mark.parametrize(
    "start_epoch, warmup, steps_per_decay, learning_rate",
    [
        (1, 0, 16, 1), (2, 10, 2, 3e-2), (3, 30, 50, 8e-4)
    ])
def test_linear_schedule_with_continue_from_epoch(start_epoch, warmup, steps_per_decay, learning_rate):
    """Make sure the optimiser restarts the schedule from the correct point when resuming training
    from a given epoch"""

    config = TestConfig(
        continue_training_from_epoch=start_epoch,
        epochs=10,
        learning_rate_function="Linear",
        lr_warmup_steps=warmup,
        lr_steps_per_decay_update=steps_per_decay,
        batches_per_step=10,
        learning_rate=learning_rate)

    iteration = Iteration(
        config,
        steps_per_epoch=20,
        writer=None,
        recording_steps=1)

    def generate_step_truth():
        if warmup > 0:
            schedule = linear_schedule(0, warmup, 1, 1e-7, learning_rate)
        else:
            schedule = {}
        schedule.update(linear_schedule(warmup, iteration.total_steps, steps_per_decay, learning_rate, 1e-7))
        return schedule
    schedule = generate_step_truth()
    step_to_match = list(filter(lambda k: k <= iteration.count, schedule.keys()))[-1]
    expected = schedule[step_to_match]

    factory = LinearOptimizerFactory(config, iteration)
    lr = factory.option_values["defaultLearningRate"]
    assert lr == expected
