# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
import scipy.sparse as sp
import tensorflow as tf

from data_utils.dataset_batch_generator import (
    add_self_edges_with_dummy_values,
    set_self_edge_dummy_values_to_zero,
    decompose_sparse_adjacency,
)
from tests.utils import convert_to_dense_and_squeeze_if_needed
from utilities.constants import AdjacencyForm
import utilities.sparse_mat_ops as mat_ops


ADJACENCY_FORMS = [AdjacencyForm.DENSE, AdjacencyForm.SPARSE_TENSOR, AdjacencyForm.SPARSE_TUPLE]


@pytest.mark.parametrize("adjacency_form", ADJACENCY_FORMS)
def test_add(adjacency_form):
    a = np.array([[0, 0, 1], [0, 0, 6], [7, 0, 0]])
    diag = np.array([[0.3, 0, 0], [0, 0.4, 0], [0, 0, 0.9]])

    if adjacency_form == AdjacencyForm.SPARSE_TENSOR:
        a = tf.sparse.from_dense(tf.constant(a, dtype=tf.float32))
        diag = tf.sparse.from_dense(tf.constant(diag, dtype=tf.float32))
    if adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        a = add_self_edges_with_dummy_values(sp.csr_matrix(a))
        a_indices, a_values, a_shape = set_self_edge_dummy_values_to_zero(a)
        a = tf.constant(a_indices), tf.constant(a_values), tf.constant(a_shape)

        diag = sp.csr_matrix(diag)
        diag_indices, diag_values, diag_shape = decompose_sparse_adjacency(diag.asformat("coo"))
        diag = (tf.constant(diag_indices), tf.constant(diag_values), tf.constant(diag_shape))

    s = mat_ops.add(a, diag)
    s = convert_to_dense_and_squeeze_if_needed(s)
    np.testing.assert_almost_equal(s, np.array([[0.3, 0, 1], [0, 0.4, 6], [7, 0, 0.9]]))


@pytest.mark.parametrize("adjacency_form", ADJACENCY_FORMS)
def test_eye(adjacency_form):
    eye = mat_ops.eye(4, tf.float32, adjacency_form)

    eye = convert_to_dense_and_squeeze_if_needed(eye)
    np.testing.assert_equal(eye, np.eye(4, dtype=np.float32))


@pytest.mark.parametrize("adjacency_form", ADJACENCY_FORMS)
def test_diag(adjacency_form):
    x = tf.constant([1, 2, 3])
    d = mat_ops.diag(x, adjacency_form)

    d = convert_to_dense_and_squeeze_if_needed(d)
    np.testing.assert_equal(d, np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]))


@pytest.mark.parametrize("adjacency_form", ADJACENCY_FORMS)
def test_diag_matmul(adjacency_form):
    a_diag_vec = tf.constant([2, 3, 4])
    sp_b = tf.constant([[1, 0, 0], [0, 0, 6], [7, 0, 0]])

    if adjacency_form in [AdjacencyForm.SPARSE_TENSOR, AdjacencyForm.SPARSE_TUPLE]:
        sp_b = tf.sparse.from_dense(sp_b)
    if adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        sp_b = (sp_b.indices, sp_b.values, sp_b.shape)

    d = mat_ops.diag_matmul(a_diag_vec, sp_b)
    d = convert_to_dense_and_squeeze_if_needed(d)
    np.testing.assert_equal(d, np.array([[2, 0, 0], [0, 0, 18], [28, 0, 0]]))


@pytest.mark.parametrize("adjacency_form", ADJACENCY_FORMS)
def test_diag_part(adjacency_form):
    x = tf.constant([[1, 0, 2], [0, 3, 4], [5, 0, 6]])
    expected_result = np.array([1, 3, 6])
    if adjacency_form == AdjacencyForm.SPARSE_TENSOR:
        x = tf.sparse.from_dense(x)
    if adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        x = sp.csr_matrix(x)
        x_indices, x_values, x_shape = decompose_sparse_adjacency(x.asformat("coo"))
        x = (tf.constant(x_indices), tf.constant(x_values), tf.constant(x_shape))

    d = mat_ops.diag_part(x)
    np.testing.assert_equal(d, expected_result)


@pytest.mark.parametrize("adjacency_form", ADJACENCY_FORMS)
@pytest.mark.parametrize(
    "adjacency, expected_row_sum",
    [
        (tf.constant([[1, 0, 3], [1, 0, 6], [7, 0, 9]]), np.array([4, 7, 16])),
        (tf.constant([[0, 0, 0], [1, 2, 3], [4, 0, 0]]), np.array([0, 6, 4])),
    ],
)
def test_reduce_sum_rows(adjacency_form, adjacency, expected_row_sum):
    if adjacency_form in [AdjacencyForm.SPARSE_TENSOR, AdjacencyForm.SPARSE_TUPLE]:
        adjacency = tf.sparse.from_dense(adjacency)
    if adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        adjacency = (adjacency.indices, adjacency.values, adjacency.shape)

    s = mat_ops.reduce_sum_rows(adjacency)
    np.testing.assert_equal(s, expected_row_sum)


@pytest.mark.parametrize("adjacency_form", ADJACENCY_FORMS)
def test_scale_by_constant(adjacency_form):
    a = tf.constant([[1, 0, 3], [1, 0, 6], [7, 0, 9]], dtype=tf.float32)
    c = 0.5
    if adjacency_form in [AdjacencyForm.SPARSE_TENSOR, AdjacencyForm.SPARSE_TUPLE]:
        a = tf.sparse.from_dense(a)
    if adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        a = (a.indices, a.values, a.shape)

    s = mat_ops.scale_by_constant(a, c)
    s = convert_to_dense_and_squeeze_if_needed(s)
    np.testing.assert_equal(s, np.array([[0.5, 0, 1.5], [0.5, 0, 3], [3.5, 0, 4.5]]))


@pytest.mark.parametrize("adjacency_form", ADJACENCY_FORMS)
@pytest.mark.parametrize(
    "sp_a, b, expected_result",
    [
        (
            tf.constant([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=tf.float64),
            tf.constant([[1, 1, 2], [3, 0.5, 4], [5, 0.1, 0.2]], dtype=tf.float64),
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ),
        (
            tf.constant([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=tf.float64),
            tf.constant([[1, 1, 2], [3, 0.5, 4], [5, 0.1, 0.2]], dtype=tf.float64),
            np.array([[5, 0.1, 0.2], [0, 0, 0], [0, 0, 0]]),
        ),
        (
            tf.constant([[0, 0, 3], [0, 0, 3], [0, 0, 3]], dtype=tf.float64),
            tf.constant([[1, 1, 2], [3, 0.5, 4], [5, 0.1, 0.2]], dtype=tf.float64),
            np.array([[15, 0.3, 0.6], [15, 0.3, 0.6], [15, 0.3, 0.6]]),
        ),
        (
            tf.constant([[3, 3, 3], [0, 0, 0], [0, 0, 0]], dtype=tf.float64),
            tf.constant([[1, 1, 2], [3, 0.5, 4], [5, 0.1, 0.2]], dtype=tf.float64),
            np.array([[27, 4.8, 18.6], [0, 0, 0], [0, 0, 0]]),
        ),
        (
            tf.constant([[1, 0, 0], [0, 2.4, 0], [0, 0, 5.6]], dtype=tf.float64),
            tf.constant([[1, 1, 2], [3, 0.5, 4], [5, 0.1, 0.2]], dtype=tf.float64),
            np.array([[1, 1, 2], [7.2, 1.2, 9.6], [28, 0.56, 1.12]]),
        ),
        (
            tf.constant([[1, 0, 3], [1, 0, 6], [7, 0, 9]], dtype=tf.float64),
            tf.constant([[1, 1, 2], [3, 0.5, 4], [5, 0.1, 0.2]], dtype=tf.float64),
            np.array([[16, 1.3, 2.6], [31, 1.6, 3.2], [52, 7.9, 15.8]]),
        ),
    ],
)
def test_sp_dense_matmul(sp_a, b, expected_result, adjacency_form):
    if adjacency_form in [AdjacencyForm.SPARSE_TENSOR, AdjacencyForm.SPARSE_TUPLE]:
        sp_a = tf.sparse.from_dense(sp_a)
    if adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        sp_a = (sp_a.indices, sp_a.values, sp_a.shape)

    p = mat_ops.sp_dense_matmul(sp_a, b)
    np.testing.assert_almost_equal(p, expected_result)
