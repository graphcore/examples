# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from enum import Enum

"""
MASKED_LABEL_VALUE: The value replace nodes that are masked in the labels.
"""
MASKED_LABEL_VALUE = -1


class Task(Enum):
    BINARY_MULTI_LABEL_CLASSIFICATION = 1
    MULTI_CLASS_CLASSIFICATION = 2
