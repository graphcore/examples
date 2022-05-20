# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from pathlib import Path

import numpy as np


def get_app_root_dir():
    return Path(__file__).parent.parent.resolve()


def assert_equal_adjacency(adjacency, expected_adjacency, is_sparse_adjacency):
    if is_sparse_adjacency:
        adjacency_row = adjacency[0][:, 0]
        adjacency_col = adjacency[0][:, 1]
        adjacency_data = adjacency[1]
        adjacency_shape = adjacency[2]
        expected_row = expected_adjacency.row
        expected_col = expected_adjacency.col
        expected_data = expected_adjacency.data
        expected_shape = expected_adjacency.shape
        np.testing.assert_array_equal(adjacency_row, expected_row)
        np.testing.assert_array_equal(adjacency_col, expected_col)
        np.testing.assert_array_equal(adjacency_data, expected_data)
        np.testing.assert_array_equal(adjacency_shape, expected_shape)
    else:
        expected_adjacency = expected_adjacency.tocsr().toarray()
        np.testing.assert_array_equal(adjacency, expected_adjacency)
