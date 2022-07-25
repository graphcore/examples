# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Mapping
import numpy as np


def _linear_schedule(start: int, end: int, interval: int, low: float, high: float) -> Mapping[int, float]:
    update_steps = np.arange(start, end+1, interval).astype(np.uint32)
    updates = np.linspace(low, high, len(update_steps))
    return dict(zip(update_steps, updates))


def linear_schedule(total_steps: int, minimum: float, maximum: float, warmup_prop: float = 0):
    """
    Learning rate schedule with warm up and warm down.
    """
    schedule = {}
    warmup_steps = int(total_steps*warmup_prop)
    if warmup_steps > 0:
        schedule.update(
            _linear_schedule(0,
                             warmup_steps,
                             1,
                             minimum,
                             maximum))

    schedule.update(
        _linear_schedule(warmup_steps,
                         total_steps,
                         1,
                         maximum,
                         minimum))
    return schedule
