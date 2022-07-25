# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from utilities.constants import AdjacencyForm


def edge_list_to_sparse_adj(edge_list, num_nodes):
    adj = sp.csr_matrix(
            (
                np.ones((edge_list.shape[0]), dtype=np.float32),
                (edge_list[:, 0], edge_list[:, 1])
            ),
            shape=(num_nodes, num_nodes))
    return adj


def get_app_root_dir():
    return Path(__file__).parent.parent.resolve()


def assert_equal_adjacency(adjacency, expected_adjacency, adjacency_form):
    if adjacency_form == AdjacencyForm.SPARSE_TENSOR:
        assert isinstance(adjacency, tf.sparse.SparseTensor)
        expected_indices = np.array([
            expected_adjacency.row,
            expected_adjacency.col
        ]).transpose()
        expected_data = expected_adjacency.data
        expected_shape = expected_adjacency.shape
        np.testing.assert_array_equal(adjacency.indices, expected_indices)
        np.testing.assert_array_equal(adjacency.values, expected_data)
        np.testing.assert_array_equal(adjacency.dense_shape, expected_shape)
    elif adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        assert isinstance(adjacency, tuple)
        expected_adjacency = expected_adjacency.tocsr().toarray()
        adjacency = tuple(tf.squeeze(a) for a in adjacency)
        indices = adjacency[0].numpy()
        values = adjacency[1].numpy()
        shape = expected_adjacency.shape
        # Remove fake node
        padding = np.where(indices[:, 0] == shape[0])[0][0]
        indices = indices[:padding, :]
        values = values[:padding]

        if values.dtype == np.float16:
            values = values.astype(np.float32)
        adjacency_mat = sp.coo_matrix((values, (indices[:, 0], indices[:, 1])), shape).toarray()
        np.testing.assert_array_equal(adjacency_mat, expected_adjacency)
    else:
        expected_adjacency = expected_adjacency.tocsr().toarray()
        np.testing.assert_array_equal(adjacency, expected_adjacency)


def convert_to_dense_and_squeeze_if_needed(x):
    if isinstance(x, tf.SparseTensor):
        x = tf.sparse.to_dense(x)
    elif isinstance(x, tuple):
        x = tf.SparseTensor(tf.cast(x[0], tf.int64), x[1], tf.cast(x[2], tf.int64))
        x = tf.sparse.to_dense(tf.sparse.reorder(x))
    return x
