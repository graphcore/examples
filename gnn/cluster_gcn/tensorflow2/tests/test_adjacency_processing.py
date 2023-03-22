# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
import scipy.sparse as sp
import tensorflow as tf

from data_utils.dataset_batch_generator import decompose_sparse_adjacency
from model.adjacency_processing import AdjacencyProcessing
from tests.utils import convert_to_dense_and_squeeze_if_needed
from utilities.constants import AdjacencyForm


ADJACENCY_FORMS = [AdjacencyForm.DENSE, AdjacencyForm.SPARSE_TENSOR, AdjacencyForm.SPARSE_TUPLE]


@pytest.mark.parametrize("adjacency_form", ADJACENCY_FORMS)
@pytest.mark.parametrize(
    "params,expected_output",
    [
        (
            dict(transform_mode="normalised"),
            np.array([[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        ),
        (
            dict(transform_mode="normalised_regularised", regularisation=0.001),
            np.array([[0.001, 0.5, 0.5], [1.0, 0.001, 0.0], [1.0, 0.0, 0.001]], dtype=np.float32),
        ),
        (
            dict(transform_mode="self_connections_scaled_by_degree"),
            np.array([[0.33333333, 0.33333333, 0.33333333], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5]], dtype=np.float32),
        ),
        (
            dict(transform_mode="normalised_regularised_self_connections_scaled_by_degree", regularisation=0.001),
            np.array([[1.001 / 3, 0.5 / 3, 0.5 / 3], [0.5, 0.5005, 0.0], [0.5, 0.0, 0.5005]], dtype=np.float32),
        ),
        (
            dict(transform_mode="self_connections_scaled_by_degree_with_diagonal_enhancement", diag_lambda=1.0),
            np.array([[0.66666667, 0.33333333, 0.33333333], [0.5, 1.0, 0.0], [0.5, 0.0, 1.0]], dtype=np.float32),
        ),
    ],
)
def test_adjacency_processing(params, adjacency_form, expected_output):
    adjacency = [[False, True, True], [True, False, False], [True, False, False]]
    adjacency = tf.constant(adjacency, dtype=tf.float32)
    if adjacency_form == AdjacencyForm.DENSE:
        num_nodes = None
    elif adjacency_form == AdjacencyForm.SPARSE_TENSOR:
        num_nodes = 3
        adjacency = tf.sparse.from_dense(adjacency)
    elif adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        num_nodes = 3
        adjacency = sp.csr_matrix(adjacency)
        # Add self-loops with zero values as it is done in the batch generator.
        adjacency += -1 * sp.eye(adjacency.shape[0])
        indices, values, shape = decompose_sparse_adjacency(adjacency.asformat("coo"))
        values[np.where(values == -1)] = 0
        adjacency = (indices, values)

    adjacency_processing_layer = AdjacencyProcessing(num_nodes, **params, adjacency_form=adjacency_form)
    output = adjacency_processing_layer(adjacency)

    output = convert_to_dense_and_squeeze_if_needed(output)
    np.testing.assert_almost_equal(output, expected_output)
