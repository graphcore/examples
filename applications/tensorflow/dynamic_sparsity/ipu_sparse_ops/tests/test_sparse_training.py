# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import os
import sys
import numpy as np
import pytest
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cwd, '..', '..'))


def assert_equal(a, b):
    assert np.array_equal(a, b, equal_nan=True)


class TestBuildAndRun(SubProcessChecker):

    @pytest.mark.category1
    def test_pruning_simple(self):
        from ipu_sparse_ops import sparse_training
        rows = np.array([1, 2, 3, 4, 5])
        cols = np.array([11, 12, 13, 14, 15])
        values = np.array([10, 20, 30, 40, 50])
        slot_values = None
        t, m = sparse_training.prune_bottom_k_weights(rows, cols, values, slot_values, 2, "test")
        assert_equal(t[0], [3, 4, 5])
        assert_equal(t[1], [13, 14, 15])
        assert_equal(t[2], [30, 40, 50])
        assert m == {}

    @pytest.mark.category1
    def test_pruning_momentum(self):
        from ipu_sparse_ops import sparse_training
        rows = np.array([1, 2, 3, 4, 5])
        cols = np.array([11, 12, 13, 14, 15])
        values = np.array([10, 20, -3, 40, 5])
        slot_values = {'momentum': np.array([0.1, 0.2, 0.3, 0.4, 0.5])}
        t, m = sparse_training.prune_bottom_k_weights(rows, cols, values, slot_values, 3, "test")
        assert_equal(t[0], [2, 4])
        assert_equal(t[1], [12, 14])
        assert_equal(t[2], [20, 40])
        assert_equal(m['momentum'], [0.2, 0.4])

    @pytest.mark.category1
    def test_regrow_rigl(self):
        from ipu_sparse_ops import sparse, sparse_training
        dense = np.array(
            [[0.1, 0.2],
             [0.3, 0.4]])
        g = np.array(
            [[1, 1, 1, 1, 1000],  # largest grad in this row is at index (0, 4)
             [1, 1, 1000, 1, 1]])  # largest grad in this row is at index (1, 2)
        a = sparse.triplets_from_dense(dense)
        t = sparse_training.regrow_rigl(a, g, sparse_training.zero_values_generator, 2, "test")
        # Coords of largest grads are (0, 4) and (1, 2):
        assert_equal(t[0], [0, 1])  # row indices
        assert_equal(t[1], [4, 2])  # col indices
        assert_equal(t[2], [0, 0])  # New values are 0 from the generator

    @pytest.mark.category1
    def test_regrow_rigl_zero_grad(self):
        from ipu_sparse_ops import sparse, sparse_training
        dense = np.array(
            [[0.1, 0.2],
             [0.3, 0.4]])
        g = np.array(
            [[1, 1, 0, 0.1, 0],  # largest grad in this row is at index (0, 3)
             [1, 1, 0, 0, 0]])
        a = sparse.triplets_from_dense(dense)
        t = sparse_training.regrow_rigl(a, g, sparse_training.zero_values_generator, 2, "test")
        print(t)
        # No guarantee about index of second value because we don't use stable sort in regrow_rigl
        # so only test the first index:
        assert t[0][0] == 0
        assert t[1][0] == 3
        assert_equal(t[2], [0, 0])  # New values are 0 from the generator

    @pytest.mark.category1
    def test_zeros(self):
        from ipu_sparse_ops import sparse_training
        assert_equal([0, 0, 0, 0], sparse_training.zero_values_generator(4))
        assert_equal([0], sparse_training.zero_values_generator())

    @pytest.mark.category1
    def test_join(self):
        from ipu_sparse_ops import sparse_training
        a = (
            np.array([1, 2, 3]),
            np.array([2, 4, 6]),
            np.array([0.1, 0.2, 0.3])
        )
        b = (
            np.array([4, 5, 6]),
            np.array([8, 10, 12]),
            np.array([0.4, 0.5, 0.6])
        )
        m = {'momentum': np.array([1.4, 1.5, 1.6])}
        g, m = sparse_training.join_triplets(a, b, m, 3)
        assert_equal(g[0], [1, 2, 3, 4, 5, 6])
        assert_equal(g[1], [2, 4, 6, 8, 10, 12])
        assert_equal(g[2], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        assert_equal(m['momentum'], [1.4, 1.5, 1.6, 0, 0, 0])

    @pytest.mark.category1
    def test_join_no_momentum(self):
        from ipu_sparse_ops import sparse_training
        a = (
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([0.1, 0.2, 0.3])
        )
        b = (
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([0.4, 0.5, 0.6])
        )
        g, m = sparse_training.join_triplets(a, b, None, 3)
        assert_equal(g[0], [1, 2, 3, 1, 2, 3])
        assert_equal(g[1], [4, 5, 6, 4, 5, 6])
        assert_equal(g[2], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        assert m == {}
