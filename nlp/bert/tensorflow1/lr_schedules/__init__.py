# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
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

from . import custom, natural_exponential, polynomial_decay, cosine_decay
import logging

from log import logger


def _warmup_steps(warmup_val, total_steps):
    # If the warmup is an integer it is going to be read a steps
    if int(warmup_val) == warmup_val:
        warmup_steps = warmup_val
    # If the warmup is a float in between 0 and 1 it is going to be read as a ratio
    elif 0.0 <= warmup_val < 1.0:
        warmup_steps = int(total_steps * warmup_val)
    # Any other combination is going to raise an error
    else:
        raise ValueError(f"Warmup can be either an integer or a float between 0 and 1. The value {warmup_val} is not acceptable")
    logger.info(f"Warmup period set to {warmup_steps} steps")
    return warmup_steps


def make_lr_schedule(name, opts, total_steps):
    """Create a learning rate decay schedule.
    Args:
        name: name of the learning rate schedule: "custom", "natural_exponential", "polynomial_decay"
        opts: a dictionary of options specific to a given schedule.
        total_steps: total number of steps in the training session.
    """

    if name == "custom":
        logger.info("Using Custom Learning Rate")
        return custom.LearningRate(
            opts["base_learning_rate"], opts["lr_schedule_by_step"]
        )

    elif name == "natural_exponential":
        logger.info("Using Natural Exponential Learning Rate")
        return natural_exponential.LearningRate(
            opts["base_learning_rate"],
            _warmup_steps(opts['warmup'], total_steps),
            opts["decay_steps"],
            opts["decay_rate"],
            total_steps
        )

    elif name == "polynomial_decay":
        logger.info("Using Polynomial Learning Rate")
        return polynomial_decay.LearningRate(
            opts["base_learning_rate"],
            _warmup_steps(opts['warmup'], total_steps),
            total_steps,
            opts.get("lr_power", 1)
        )
    elif name == "cosine_decay":
        logger.info(f"Using Cosine Learning Rate decay")
        return cosine_decay.LearningRate(
            opts["base_learning_rate"],
            _warmup_steps(opts['warmup'], total_steps),
            total_steps,
            opts.get("lr_power", 1)
        )
    else:
        raise ValueError("Learning Rate not implemented.")
