# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import MutableMapping
from typing import Mapping, Optional, Tuple
import os

import numpy as np
import popxl
from popxl import ReplicaGrouping
from popxl.context import get_current_context
from popxl.tensor import Variable

from config import GPTConfig


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


def tp2d_replica_groups(
    config: GPTConfig,
) -> Tuple[ReplicaGrouping, ReplicaGrouping, ReplicaGrouping, ReplicaGrouping]:
    """Output the communication replica groupings for TP1, TP2, TP-All and DP.
    The variable partition replica groups are the transpose of each corresponding group.
    """

    ir = get_current_context().graph.ir
    dp = config.execution.data_parallel
    tp1 = config.execution.tensor_parallel_1
    tp2 = config.execution.tensor_parallel_2

    assert (
        ir.replication_factor == dp * tp1 * tp2
    ), f"replication_factor: {ir.replication_factor}, dp: {dp}, tp1: {tp1}, tp2: {tp2}"

    rg_tp1_ = ir.replica_grouping(group_size=tp1, stride=tp2)  # Outermost
    rg_tp2_ = ir.replica_grouping(group_size=tp2, stride=1)  # Innermost
    rg_tp_all_ = ir.replica_grouping(group_size=tp1 * tp2, stride=1)
    rg_dp = rg_tp_all_.transpose()

    return rg_tp1_, rg_tp2_, rg_tp_all_, rg_dp
