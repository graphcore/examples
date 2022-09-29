# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np


def shard(x: np.ndarray, n_shards: int, axis: int) -> np.array:
    """Shard array along a given axis"""

    if n_shards == 1:
        return x

    if axis < 0:
        axis = len(x.shape) + axis

    return np.ascontiguousarray(
        np.concatenate(np.split(x[np.newaxis, ...], n_shards, axis=axis+1))
    )


def repeat(x: np.ndarray, n: int, axis: int = 0) -> np.array:
    """Repeat array along new axis inserted at position `axis`"""
    return np.repeat(np.expand_dims(x, axis), n, axis=axis)
