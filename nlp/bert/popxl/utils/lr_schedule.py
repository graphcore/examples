# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Mapping
import numpy as np


__all__ = ["linear_schedule"]


def _linear_schedule(start: int, end: int, interval: int, low: float, high: float) -> Mapping[int, float]:
    update_steps = np.arange(start, end + 1, interval).astype(np.uint32)
    updates = np.linspace(low, high, len(update_steps))
    return dict(zip(update_steps, updates))


def linear_schedule(total_steps: int, minimum: float, maximum: float, warmup_prop: float = 0):
    """Learning rate schedule with linear warm up and linear warm down.

    Linearly increase from `minimum` to `maximum` for `total_steps*warmup_prop` steps.
    Then for the remaining steps linearly decrease from `maximum` to `minimum`.

    Returns a dict that maps step to learning rate.
    """
    schedule = {}
    warmup_steps = int(total_steps * warmup_prop)
    if warmup_steps > 0:
        schedule.update(_linear_schedule(0, warmup_steps, 1, minimum, maximum))

    schedule.update(_linear_schedule(warmup_steps, total_steps, 1, maximum, minimum))
    return schedule
