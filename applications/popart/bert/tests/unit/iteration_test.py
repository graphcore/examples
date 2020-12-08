# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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
import random
import math
import numpy as np
import itertools
from typing import NamedTuple
from collections import defaultdict
from bert import Iteration


class MockArgs(NamedTuple):
    start_epoch: int = 0
    epochs: int = 10
    continue_training_from_epoch: int = 0
    gradient_accumulation_factor: int = 1
    replication_factor: int = 1
    epochs_per_save: int = 1
    steps_per_log: int = 1
    batch_size: int = 1
    inference_lm_perplexity: bool = False
    inference: bool = False
    task: str = "SQUAD"
    squad_lr_scale: int = 1


class MockWriter():
    def __init__(self):
        self.scalars = defaultdict(dict)

    def add_scalar(self, key, val, step_num):
        self.scalars[key][step_num] = val


@pytest.mark.category1
@pytest.mark.parametrize(
    "task, epochs, seed, exponent_loss, exponent_acc",
    list(itertools.product(
        ("PRETRAINING", "SQUAD"),
        range(1, 6, 2),
        [1980, 1981],
        [-0.05, -0.1],
        [0.05, 0.1]))
)
def test_iteration_stats(task, epochs, seed, exponent_loss, exponent_acc):
    """Assuming aggregate-metrics-over-steps is 1, the data recorded by the iteration and sent to the
    TB logger should exactly match the true data. Here we mock everything out, bar the iteration class
    itself and ensure this is the case."""

    random.seed(seed)

    total_data_length = 5000
    batches_per_step = 500
    recording_steps = 1  # --aggregate-metrics-over-steps

    dataset_length = int(total_data_length / batches_per_step)  # len(dataset)
    if total_data_length % batches_per_step != 0:
        raise ValueError(
            "Dataset not divisible by bps, not supported in this test.")

    args = MockArgs(**{"epochs": epochs, "task": task})
    mock_writer = MockWriter()

    def generate_known_exp_curve(task, epochs, dataset_length, gamma):
        # Force the losses to be slightly different in the case of two losses, to make sure we account
        # for expected differences  between them
        multipliers = (0.9, 1.05) if task == "PRETRAINING" else (1,)
        return {step: [m*math.exp(gamma*step) for m in multipliers] for step in range(int(epochs*dataset_length))}

    step_losses = generate_known_exp_curve(
        task, epochs, dataset_length, exponent_loss)
    step_accuracies = generate_known_exp_curve(
        task, epochs, dataset_length, exponent_acc)

    def generate_step_result(step_num):
        duration = random.random()
        hw_cycles = random.randint(1000, 2000)
        return duration, hw_cycles, step_losses[step_num], step_accuracies[step_num]

    def mock_stats_fn(loss, accuracy):
        return loss, accuracy

    iteration = Iteration(args, batches_per_step,
                          dataset_length, mock_writer, recording_steps)
    iteration.stats_fn = mock_stats_fn

    epoch_steps = []
    for iteration.epoch in range(args.start_epoch, args.epochs):
        epoch_steps.append(iteration.count)
        for data in range(dataset_length):
            step_result = generate_step_result(iteration.count)
            iteration.add_stats(*step_result)
            iteration.count += 1

    if task == "PRETRAINING":
        loss_keys = ["loss/MLM", "loss/NSP"]
        acc_keys = ["accuracy/MLM", "accuracy/NSP"]
    else:
        loss_keys = ["loss"]
        acc_keys = ["accuracy"]

    def compare_against_true_curve(keys, true_values, writer):
        for key_index, loss_key in enumerate(keys):
            for step_num, loss in writer.scalars[loss_key].items():
                isclose = np.isclose(true_values[step_num][key_index], loss)
                assert isclose

    compare_against_true_curve(loss_keys, step_losses, mock_writer)
    compare_against_true_curve(acc_keys, step_accuracies, mock_writer)


if __name__ == "__main__":
    pytest.main(args=[__file__, '-vv', '-s'])
