# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf
import pytest
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cwd, '..', '..'))


def assert_equal(a, b):
    assert np.array_equal(a, b, equal_nan=True)


class TestBuildAndRun(SubProcessChecker):

    @pytest.mark.category1
    def test_conversions(self):
        from ipu_sparse_ops import sparse
        m = np.array([[10, 0], [0, 20]])
        t = sparse.triplets_from_dense(m)
        assert_equal(t[0], [0, 1])
        assert_equal(t[1], [0, 1])
        assert_equal(t[2], [10, 20])
        spec = sparse.matmul_spec_from_max(2, [1, 2], 2, tf.float32)
        n = sparse.dense_from_triplets(spec, *t)
        assert_equal(n, m)
        o = sparse.mask_from_triplets(spec, *t)
        assert_equal(o, np.array([[1, 0], [0, 1]]))

    @pytest.mark.category1
    def test_random_indices(self):
        from ipu_sparse_ops import sparse
        spec = sparse.matmul_spec_from_max(10, [1, 20], 10, tf.float32)
        r, c = sparse.random_indices(spec, None)
        print(f"r,c:\n{r}\n{c}")
        assert len(r) == 10
        assert len(r) == len(c)
        assert np.max(r) < spec.input_size
        assert np.max(c) < spec.output_size
        assert np.min(r) >= 0
        assert np.min(c) >= 0

    @pytest.mark.category1
    def test_disjoint_random_indices(self):
        from ipu_sparse_ops import sparse
        spec = sparse.matmul_spec_from_max(10, [1, 20], 10, tf.float32)
        ta, tb = sparse.disjoint_random_indices(spec, size_a=10, size_b=5,
                                                indices_initialiser_gen=None)
        print(f"r,c:\n{ta}\n{tb}")
        assert len(ta[0]) == 10
        assert len(tb[0]) == 5
        # Check the result is disjoint by checking uniqueness of ravelled indices:
        shape = (spec.input_size, spec.output_size)
        a_flat = np.ravel_multi_index((ta[0], ta[1]), shape)
        b_flat = np.ravel_multi_index((tb[0], tb[1]), shape)
        all_flat = np.concatenate((a_flat, b_flat), axis=0)
        unique = np.unique(all_flat)
        assert len(unique) == len(all_flat)
        # Check we get an error if total number of indices is too large to be unique:
        with pytest.raises(ValueError):
            ta, tb = sparse.disjoint_random_indices(spec, size_a=150, size_b=100,
                                                    indices_initialiser_gen=None)
