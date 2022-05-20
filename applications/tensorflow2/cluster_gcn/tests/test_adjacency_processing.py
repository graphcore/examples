# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
import tensorflow as tf

from model.adjacency_processing import AdjacencyProcessing


@pytest.mark.parametrize('params,expected_output', [
    (
            dict(transform_mode="normalised"),
            np.array([[0., 0.5, 0.5],
                      [1., 0., 0.],
                      [1., 0., 0.]], dtype=np.float32)
    ),
    (
            dict(transform_mode="normalised_regularised", regularisation=0.001),
            np.array([[0.001, 0.5, 0.5],
                      [1., 0.001, 0.],
                      [1., 0., 0.001]], dtype=np.float32)
    ),

    (
        dict(transform_mode="self_connections_scaled_by_degree"),
        np.array([[0.33333333, 0.33333333, 0.33333333],
                  [0.5, 0.5, 0.],
                  [0.5, 0., 0.5]], dtype=np.float32)
    ),
    (
        dict(transform_mode="normalised_regularised_self_connections_scaled_by_degree", regularisation=0.001),
        np.array([[1.001/3, 0.5/3, 0.5/3],
                  [0.5, 0.5005, 0.],
                  [0.5, 0., 0.5005]], dtype=np.float32)
    ),
    (
        dict(transform_mode="self_connections_scaled_by_degree_with_diagonal_enhancement",
             diag_lambda=1.0),
        np.array([[0.66666667, 0.33333333, 0.33333333],
                  [0.5, 1., 0.],
                  [0.5, 0., 1.]], dtype=np.float32)
    )
])
def test_normalised_regularised(params, expected_output):
    adjacency = [[False, True, True], [True, False, False], [True, False, False]]
    adjacency = tf.constant(adjacency, dtype=tf.float32)
    diagonal_enhancement = AdjacencyProcessing(**params)
    output_from_layer = diagonal_enhancement(adjacency)
    np.testing.assert_almost_equal(output_from_layer, expected_output)
