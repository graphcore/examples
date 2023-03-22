# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from enum import Enum, auto

CLUSTERING_CACHE_EXT = ".npy"

"""
MASKED_LABEL_VALUE: The value replace nodes that are masked in the labels.
"""
MASKED_LABEL_VALUE = -1


class Task(Enum):
    BINARY_MULTI_LABEL_CLASSIFICATION = auto()
    MULTI_CLASS_CLASSIFICATION = auto()


class AdjacencyForm(Enum):
    DENSE = auto()
    SPARSE_TENSOR = auto()
    SPARSE_TUPLE = auto()


class GraphType(Enum):
    DIRECTED = auto()
    UNDIRECTED = auto()


class MethodMaxNodesEdges(Enum):
    """Method to estimate the maximum number of nodes/edges per batch."""

    AVERAGE = auto()
    AVERAGE_PLUS_STD = auto()
    UPPER_BOUND = auto()
