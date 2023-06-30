# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import os
from typing import Dict, Iterable, Mapping, Optional, Callable, List, Tuple

import popxl
from popxl import Tensor, Session, ReplicaGrouping, gir
import numpy as np
import popart
from config.config import GPTConfig


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


def suffix_path(path: str, suffix: str):
    """Add a suffix to a filename in a path. The suffix is affixed before the file extension"""
    path, ext = os.path.splitext(path)
    return path + suffix + ext


def replica_groups(
    config: GPTConfig,
) -> Tuple[ReplicaGrouping, ReplicaGrouping]:
    """Output the communication replica groupings for TP and DP.
    The variable partition replica groups are the transpose of each corresponding group.
    """

    ir = gir()
    dp = config.execution.data_parallel
    tp = config.execution.tensor_parallel

    assert ir.replication_factor == dp * tp, f"replication_factor: {ir.replication_factor}, dp: {dp}, tp1: {tp}"

    rg_tp = ir.replica_grouping(group_size=tp, stride=1)
    rg_dp = rg_tp.transpose()

    return rg_tp, rg_dp
