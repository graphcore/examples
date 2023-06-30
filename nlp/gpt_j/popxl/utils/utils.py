# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import Dict, Iterable, Mapping, Optional, Callable, List

import popxl
from popxl import Tensor, Session
import numpy as np
import popart
import os
import time


def shard(x: np.ndarray, n_shards: int, axis: int) -> np.array:
    """Shard array along a given axis"""
    if axis < 0:
        axis = len(x.shape) + axis

    return np.ascontiguousarray(np.concatenate(np.split(x[np.newaxis, ...], n_shards, axis=axis + 1)))


def repeat(x: np.ndarray, n: int, axis: int = 0) -> np.array:
    """Repeat array along new axis inserted at position `axis`"""
    return np.repeat(np.expand_dims(x, axis), n, axis=axis)


def tensor_parallel_input(
    input_data: np.ndarray,
    tp: int,
    rf: int,
    repeat_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
):
    """Repeat the data in `input_data` such that consecutive replicas with groupSize tp get the same data
    (optionally modified by repeat_fn)

    Take data of shape (host_loads, *data_shape)
    Output (host_loads, replication_factor, *data_shape)
    Where host_loads = device_iterations * gradient_accumulation_step

    Examples:
        a = np.arange(4)
        [0, 1, 2, 3]
        shape: (4,)

        tensor_parallel_input(a, 1, 2)  # tp = 1, dp = 2
        [[0, 1],
         [2, 3]]
        shape: (2, 2)

        tensor_parallel_input(a, 2, 2)  # tp = 2, dp = 1
        [[0, 0],
         [1, 1],
         [2, 2],
         [3, 3]]
        shape: (4, 2)

        tensor_parallel_input(a, 2, 4)  # tp = 2, dp = 2
        [[0, 0, 1, 1],
         [2, 2, 3, 3]]
        shape: (2, 4)

    Args:
        input_data (np.ndarray): Data to repeat
        tp (int): Tensor parallel replicas
        rf (int): Total Replicas
        repeat_fn (Optional[Callable[[np.ndarray, int], np.ndarray]], optional):
            Optional function to modify each repeat by. Defaults to None.

    Returns:
        data: Data repeated for DP and TP
    """
    assert tp <= rf
    assert (rf / tp).is_integer()

    data = np.expand_dims(input_data, axis=1)
    repeats: List[np.ndarray] = []
    for i in range(tp):
        repeat = data
        if repeat_fn:
            repeat = repeat_fn(repeat.copy(), i)
        repeats.append(repeat)
    data = np.concatenate(repeats, axis=1)
    return data.reshape(-1, rf, *input_data.shape[1:])


def _linear_schedule(start: int, end: int, interval: int, low: float, high: float) -> Dict[int, float]:
    update_steps = np.arange(start, end + 1, interval).astype(np.uint32)
    updates = np.linspace(low, high, len(update_steps))
    return dict(zip(update_steps, updates))


def warmup_schedule(total_steps: int, minimum: float, maximum: float, warmup_prop: float = 0) -> Dict[int, float]:
    """Learning rate schedule with linear warm up and then remains at max.

    Linearly increase from `minimum` to `maximum` for `total_steps*warmup_prop` steps.
    Then constant at the `maximum` learning rate for the remaining steps.

    Returns a dict that maps step to learning rate.
    """
    schedule = {}
    warmup_steps = int(total_steps * warmup_prop)
    if warmup_steps > 0:
        schedule.update(_linear_schedule(0, warmup_steps, 1, minimum, maximum))

    schedule.update(_linear_schedule(warmup_steps, total_steps, 1, maximum, maximum))  # maximum to maximum so constant
    return schedule


def suffix_path(path: str, suffix: str):
    """Add a suffix to a filename in a path. The suffix is affixed before the file extension"""
    path, ext = os.path.splitext(path)
    return path + suffix + ext


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
