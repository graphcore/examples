# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import tensorflow as tf
import numpy as np
import pytest
import scipy.sparse as sp
import tensorflow as tf

from data_utils.dataset_batch_generator import (
    add_self_edges_with_dummy_values,
    pad_adjacency_tuple,
    tf_dataset_generator
)
from tests.utils import assert_equal_adjacency
from utilities.constants import AdjacencyForm


def test_pad_adjacency_tuple():
    edges = np.array([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3]])
    max_num_nodes = 5  # Includes fake node.
    max_num_edges = 15

    expected_edges = np.array([[0, 0],  # Self-edge
                               [0, 1],
                               [1, 0],
                               [1, 1],  # Self-edge
                               [1, 2],
                               [2, 1],
                               [2, 2],  # Self-edge
                               [2, 3],
                               [3, 3],  # Self-edge
                               [4, 4],  # Self-edge
                               [4, 4],  # Dummy edge
                               [4, 4],  # Dummy edge
                               [4, 4],  # Dummy edge
                               [4, 4],  # Dummy edge
                               [4, 4]])  # Dummy edge
    expected_values = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    adjacency = sp.csr_matrix(
        (
            np.ones((edges.shape[0]), dtype=np.float32),
            (edges[:, 0], edges[:, 1])
        ),
        shape=(max_num_nodes, max_num_nodes)
    )
    adjacency = add_self_edges_with_dummy_values(adjacency)
    indices, values = pad_adjacency_tuple(
        adjacency, np.bool, max_num_edges, max_num_nodes)
    np.testing.assert_equal(indices, expected_edges)
    np.testing.assert_equal(values, expected_values)


@pytest.mark.parametrize("features_dtype", [np.float16, np.float32])
@pytest.mark.parametrize("labels_dtype", [np.int32])
@pytest.mark.parametrize(
    "adjacency_form",
    [
        AdjacencyForm.DENSE,
        AdjacencyForm.SPARSE_TENSOR,
        AdjacencyForm.SPARSE_TUPLE
    ]
)
@pytest.mark.parametrize("adjacency_dtype", [bool, np.float32])
@pytest.mark.parametrize("max_edges_per_batch", [13, 15])
def test_tf_dataset_generator(features_dtype,
                              labels_dtype,
                              adjacency_form,
                              adjacency_dtype,
                              max_edges_per_batch):
    clusters = [np.array([0]),
                np.array([1, 2]),
                np.array([3])]
    max_nodes_per_batch = 4
    num_sample_clusters = 2
    clusters_per_batch = 2

    labels = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1]],
                      dtype=labels_dtype)
    mask = np.array([0, 1, 1, 1])
    features = np.array([[0.01, 0.02],
                         [0.11, 0.12],
                         [0.21, 0.22],
                         [0.31, 0.32]],
                        dtype=features_dtype)

    edges = np.array([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3]])

    # Expected result after sampling clusters.
    expected_edges = np.array([[0, 1], [1, 0], [1, 2], [2, 1]])
    expected_values = [1] * max_nodes_per_batch
    expected_adj_matrix = sp.coo_matrix(
        (expected_values, expected_edges.T),
        shape=(max_nodes_per_batch, max_nodes_per_batch),
        dtype=adjacency_dtype
    )
    expected_features = np.array(
        [[0.01, 0.02],
         [0.11, 0.12],
         [0.21, 0.22],
         [0., 0.]],
        dtype=features_dtype)
    expected_labels = np.array(
        [[-1, -1, -1, -1, -1, -1],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [-1, -1, -1, -1, -1, -1]],
        dtype=labels_dtype)

    # Add fake node if needed
    if adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        max_nodes_per_batch += 1
        expected_values += [0]

    adjacency = sp.csr_matrix(
        (
            np.ones((edges.shape[0]), dtype=adjacency_dtype),
            (edges[:, 0], edges[:, 1])
        ),
        shape=(max_nodes_per_batch, max_nodes_per_batch)
    )

    dataset_generator = tf_dataset_generator(
        adjacency,
        clusters,
        features,
        labels,
        mask,
        num_sample_clusters,
        clusters_per_batch,
        max_nodes_per_batch,
        max_edges_per_batch,
        adjacency_dtype,
        adjacency_form,
        seed=3,
        deterministic=True,
    )

    first_batch = iter(dataset_generator.take(1)).next()

    assert_equal_adjacency(
        first_batch[0]["adjacency_batch"],
        expected_adj_matrix,
        adjacency_form
    )

    if adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        assert first_batch[0]["adjacency_batch"][1].dtype == adjacency_dtype
        # Remove fake node.
        features = first_batch[0]["features_batch"][0][:-1]
        labels = first_batch[1][0][:-1]
    else:
        assert first_batch[0]["adjacency_batch"].dtype == adjacency_dtype
        features = first_batch[0]["features_batch"]
        labels = first_batch[1]


    np.testing.assert_array_almost_equal(features, expected_features)
    assert features.dtype == features_dtype

    np.testing.assert_array_equal(labels, expected_labels)
    assert labels.dtype == labels_dtype
