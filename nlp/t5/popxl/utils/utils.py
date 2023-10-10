# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Dict

import numpy as np
import time


def _linear_schedule(start: int, end: int, interval: int, low: float, high: float) -> Dict[int, float]:
    update_steps = np.arange(start, end + 1, interval).astype(np.uint32)
    updates = np.linspace(low, high, len(update_steps))
    return dict(zip(update_steps, updates))


def warmup_schedule(total_steps: int, minimum: float, maximum: float, warmup_steps: int = 0) -> Dict[int, float]:
    """Learning rate schedule with linear warm up and then remains at max.

    Linearly increase from `minimum` to `maximum` for `warmup_steps` steps.
    Then constant at the `maximum` learning rate for the remaining steps.

    Returns a dict that maps step to learning rate.
    """
    schedule = {}
    if warmup_steps > 0:
        schedule.update(_linear_schedule(0, warmup_steps, 1, minimum, maximum))

    schedule.update(_linear_schedule(warmup_steps, total_steps, 1, maximum, maximum))  # maximum to maximum so constant
    return schedule


class SimpleTimer:
    def __init__(self):
        self._start = None

    def start(self):
        self._start = time.perf_counter()

    def stop(self):
        self.elapsed = time.perf_counter() - self._start
        self._start = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_args):
        self.stop()
