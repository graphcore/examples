# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest

from data_utils.utils import construct_adj, sample_mask


@pytest.mark.parametrize(
    "expected_full_adj_matrix, directed_graph",
    [
        (
            np.array(
                [[0, 1, 0, 1, 1, 0],
                 [1, 0, 1, 0, 0, 0],
                 [0, 1, 0, 0, 0, 1],
                 [1, 0, 0, 0, 1, 0],
                 [1, 0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0, 0]]
            ),
            False
        ),
        (
            np.array(
                [[0, 1, 0, 1, 1, 0],
                 [0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]
            ),
            True
        ),
    ]
)
def test_construct_adj(expected_full_adj_matrix, directed_graph):
    edge_list = np.array([[0, 4],
                          [0, 3],
                          [0, 1],
                          [3, 4],
                          [1, 2],
                          [2, 5]])
    sparse_adj = construct_adj(edge_list,
                               len(expected_full_adj_matrix),
                               directed_graph)
    adj = sparse_adj.toarray()
    np.testing.assert_array_equal(adj, expected_full_adj_matrix)


def test_sample_mask():
    idx = np.array([1, 3, 4])
    num_nodes = 5
    expected_mask = np.array([0, 1, 0, 1, 1])

    mask = sample_mask(idx, num_nodes)
    np.testing.assert_array_equal(mask, expected_mask)
