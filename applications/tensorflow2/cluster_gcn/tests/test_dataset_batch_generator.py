# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import scipy.sparse as sp

from data_utils.dataset_batch_generator import tf_dataset_generator
from tests.utils import assert_equal_adjacency


def test_tf_dataset_generator():
    clusters = [np.array([0]),
                np.array([1, 2]),
                np.array([3])]
    max_nodes_per_batch = 4
    num_sample_clusters = 2
    clusters_per_batch = 2

    labels = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1]])
    mask = np.array([0, 1, 1, 1])
    features = np.array([[0.01, 0.02],
                         [0.11, 0.12],
                         [0.21, 0.22],
                         [0.31, 0.32]])

    edges = np.array([[0, 1], [1, 2], [2, 3]])
    adjacency = sp.csr_matrix(
        (
            np.ones((edges.shape[0]), dtype=np.float32),
            (edges[:, 0], edges[:, 1])
        ),
        shape=(4, 4))
    adjacency += adjacency.transpose()

    dataset_generator = tf_dataset_generator(
        adjacency,
        clusters,
        features,
        labels,
        mask,
        num_sample_clusters,
        clusters_per_batch,
        max_nodes_per_batch,
        seed=5
    )

    first_batch = list(dataset_generator.take(1).as_numpy_iterator())

    expected_edges = np.array([[0, 1],
                               [1, 0],
                               [1, 2],
                               [2, 1]])
    expected_adj_matrix = sp.coo_matrix(
        ([True] * max_nodes_per_batch, expected_edges.T),
        shape=(max_nodes_per_batch, max_nodes_per_batch),
        dtype=bool
    )
    expected_features = np.array([[0.01, 0.02],
                                  [0.11, 0.12],
                                  [0.21, 0.22],
                                  [0., 0.]])
    expected_labels = np.array(
        [[-1, -1, -1, -1, -1, -1],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [-1, -1, -1, -1, -1, -1]])

    assert_equal_adjacency(
        first_batch[0][0]["adjacency"],
        expected_adj_matrix,
        False
    )
    np.testing.assert_array_almost_equal(first_batch[0][0]["features"], expected_features)
    np.testing.assert_array_equal(first_batch[0][1],  expected_labels)
