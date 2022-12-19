# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from datetime import datetime

import numpy as np
import tensorflow as tf

from tensorflow.python.ipu import distributed
from utilities.constants import AdjacencyForm, MethodMaxNodesEdges


def get_adjacency_form(device, use_sparse_representation):
    if not use_sparse_representation:
        return AdjacencyForm.DENSE
    if device == "cpu" and use_sparse_representation:
        return AdjacencyForm.SPARSE_TENSOR
    if device == "ipu" and use_sparse_representation:
        return AdjacencyForm.SPARSE_TUPLE


def get_adjacency_dtype(device, use_sparse_representation):
    if use_sparse_representation and device == "cpu":
        # If using SparseTensor on CPU we can only allow float32.
        return np.float32
    else:
        # For dense and sparse tuple the adjacency can be a bool
        # to minimize IO.
        return np.bool


def get_method_max(method_max_str):
    if method_max_str == "average":
        return MethodMaxNodesEdges.AVERAGE
    if method_max_str == "average_plus_std":
        return MethodMaxNodesEdges.AVERAGE_PLUS_STD
    if method_max_str == "upper_bound":
        return MethodMaxNodesEdges.UPPER_BOUND


def decompose_sparse_adjacency(adjacency_coo):
    """
    Returns a sparse matrix as a tuple of (indices, values, shape).
    This is needed to feed the tf.data.Dataset from a generator.
    """
    indices = np.array([adjacency_coo.row, adjacency_coo.col], dtype=np.int32).transpose()
    values = adjacency_coo.data
    shape = np.array(adjacency_coo.shape, dtype=np.int32)
    return indices, values, shape


def get_time_now(distributed_training):
    if distributed_training:
        time_now = float(
            distributed.broadcast(tf.convert_to_tensor(value=datetime.now().timestamp(), dtype=tf.float32), 0))
    else:
        time_now = datetime.now().timestamp()
    return time_now
