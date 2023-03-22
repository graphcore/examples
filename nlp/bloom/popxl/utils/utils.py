# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import os
from typing import Mapping, MutableMapping, Optional, Tuple

import numpy as np

import popxl
from config import BloomConfig
from popxl import ReplicaGrouping
from popxl.context import get_current_context
from popxl.tensor import Variable


def suffix_path(path: str, suffix: str):
    """Add a suffix to a filename in a path. The suffix is affixed before the file extension"""
    path, ext = os.path.splitext(path)
    return path + suffix + ext


def tp2d_replica_groups(
    config: BloomConfig,
) -> Tuple[ReplicaGrouping, ReplicaGrouping, ReplicaGrouping, ReplicaGrouping]:
    """Output the communication replica groupings for TP1, TP2 and TP-All
    The variable partition replica groups are the transpose of each corresponding group.
    """

    ir = get_current_context().graph.ir
    tp1 = config.execution.tensor_parallel_1
    tp2 = config.execution.tensor_parallel_2

    assert ir.replication_factor == tp1 * tp2

    rg_tp1_ = ir.replica_grouping(group_size=tp1, stride=tp2)  # Outermost
    rg_tp2_ = ir.replica_grouping(group_size=tp2, stride=1)  # Innermost
    rg_tp_all_ = ir.replica_grouping(group_size=tp1 * tp2, stride=1)

    return rg_tp1_, rg_tp2_, rg_tp_all_, None
