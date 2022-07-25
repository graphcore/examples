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
    assert np.array_equal(a, b)


@pytest.mark.usefixtures("ipu_sparse_ops")
class TestBuildAndRun(SubProcessChecker):

    def test_block_size_and_shape(self):
        from ipu_sparse_ops import sparse
        for i in [1, 2, 4, 8, 16, 32]:
            v = [[[0.1] * i] * i] * 10
            assert sparse.block_size_from_list(v) == i
        for i in [1, 2, 4, 8, 16, 32]:
            v = [[[0.1] * 2 * i] * i]
            assert sparse.block_shape_from_list(v) == (i, 2 * i)
        # Check flattened blocks (e.g. 2x2 here) are disallowed:
        with pytest.raises(ValueError):
            sparse.block_size_from_list([[1, 2, 3, 4], [1, 2, 3, 4]])

    def test_conversions(self):
        from ipu_sparse_ops import sparse
        m = np.array([[10, 0], [0, 20]])
        t = sparse.triplets_from_dense(m)
        assert_equal(t[0], [0, 1])
        assert_equal(t[1], [0, 1])
        assert_equal(t[2], [10, 20])
        spec = sparse.matmul_spec_from_max(
            2, [1, 2], 2, block_size=1, dtype=tf.float32)
        n = sparse.dense_from_triplets(spec, *t)
        assert_equal(n, m)
        o = sparse.mask_from_triplets(spec, *t)
        assert_equal(o, np.array([[1, 0], [0, 1]]))

    def test_block_conversions(self):
        from ipu_sparse_ops import sparse
        a = np.kron([[1, 0], [1, 0]], [[1, 2], [3, 4]])
        b = np.kron([[0, 0], [1, 0]], [[4, 4], [4, 4]])
        dense = a + b
        bs = 2
        spec = sparse.matmul_spec_from_max(2*bs, [1, 2*bs], 2, block_size=bs, dtype=tf.float32)
        blocks = np.reshape([1, 2, 3, 4, 5, 6, 7, 8], [2, bs, bs])
        t = (
            [0, 1],
            [0, 0],
            blocks
        )
        n = sparse.dense_from_triplets(spec, *t)
        assert_equal(dense, n)

        # Check that mask from dense and mask from triplets
        # return the same result:
        mask_dense = np.zeros_like(dense)
        mask_dense[np.nonzero(dense)] = 1
        mask_trips = sparse.mask_from_triplets(spec, *t)
        assert_equal(mask_dense, mask_trips)

        # Check triplets from dense returns same triplets:
        td = sparse.triplets_from_dense(dense, bs)
        assert_equal(t[0], td.row_indices)
        assert_equal(t[1], td.col_indices)
        assert_equal(t[2], td.values)

    def test_random_indices(self):
        from ipu_sparse_ops import sparse
        spec = sparse.matmul_spec_from_max(
            10, [1, 20], 10, block_size=1, dtype=tf.float32)
        r, c = sparse.random_indices(spec, None)
        print(f"r,c:\n{r}\n{c}")
        assert len(r) == 10
        assert len(r) == len(c)
        assert np.max(r) < spec.input_size
        assert np.max(c) < spec.output_size
        assert np.min(r) >= 0
        assert np.min(c) >= 0

    def test_disjoint_random_indices(self):
        from ipu_sparse_ops import sparse
        spec = sparse.matmul_spec_from_max(
            10, [1, 20], 10, block_size=1, dtype=tf.float32)
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

    def test_flatten_blocks(self):
        from ipu_sparse_ops import sparse
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        print(f"shape:{values.shape}")
        spec1 = sparse.matmul_spec_from_max(
            4, [4, 4], max_non_zeros=values.size, block_size=1, dtype=tf.float32)
        bs = 2
        spec2 = sparse.matmul_spec_from_max(
            4, [4, 4], max_non_zeros=values.size//bs, block_size=bs, dtype=tf.float32)

        # Trying to flatten already flat values returns same array:
        assert_equal(values, sparse.flatten_blocks(spec1, values))

        # Block-size 1 has no effect:
        not_blocks = sparse.unflatten_blocks(spec1, values)
        assert_equal(not_blocks, values)
        # Test round trip with block size 2:
        blocks = sparse.unflatten_blocks(spec2, values)
        assert blocks.shape == (bs, bs, bs)
        values_rt = sparse.flatten_blocks(spec2, blocks)
        assert_equal(values, values_rt)

    @pytest.mark.ipus(1)
    def test_representation_round_trip_elements(self):
        from ipu_sparse_ops import sparse
        bs = 16
        block_mask = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
        mask = np.kron(block_mask, np.ones(shape=[bs, bs])).astype(int)
        n_els = np.count_nonzero(mask)
        dense = np.zeros_like(mask)
        dense[np.nonzero(mask)] = np.arange(n_els)
        opts = {"metaInfoBucketOversizeProportion": 1}
        t = sparse.triplets_from_dense(dense)
        spec = sparse.matmul_spec_from_max(
            dense.shape[1], [2, dense.shape[0]], max_non_zeros=n_els, block_size=1, dtype=tf.float32)
        r = sparse.representation_from_triplets(spec, *t, opts)
        t_rt = sparse.triplets_from_representation(spec, r, opts)
        dense_rt = sparse.dense_from_triplets(spec, *t_rt)
        assert_equal(dense, dense_rt)

    @pytest.mark.ipus(1)
    def test_representation_round_trip_blocks(self):
        from ipu_sparse_ops import sparse
        for bs in [4, 8, 16]:
            # Create a mask that describes the non-zero block structure:
            block_mask = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
            n_blocks = np.count_nonzero(block_mask)
            # From that produce an element-wise mask using a Kronecker product:
            mask = np.kron(block_mask, np.ones(shape=[bs, bs])).astype(int)
            n_els = np.count_nonzero(mask)
            # Make a dense matrix from the element-wise mask and fill with random values:
            dense = np.zeros_like(mask, dtype=np.float32)
            values = np.random.rand(n_els)
            dense[np.nonzero(mask)] = values
            # Make the spec for the sparse matmul:
            opts = {"metaInfoBucketOversizeProportion": 1}
            spec = sparse.matmul_spec_from_max(
                dense.shape[1], [2, dense.shape[0]], max_non_zeros=n_blocks, block_size=bs, dtype=tf.float32)
            # Make triplets indices from the block mask:
            t = sparse.triplets_from_dense(block_mask)
            # Then fill in triplet's values by extracting the blocks
            # from the dense matrix (this can't be done by reshaping):
            t_block = sparse.Triplets(
                t.row_indices, t.col_indices,
                sparse.blocks_at_indices(t.row_indices, t.col_indices, bs, dense)
            )
            # Convert to on device representation and back and check the
            # result is the dense matrix we sytarted with:
            r = sparse.representation_from_triplets(spec, *t_block, opts)
            t_rt = sparse.triplets_from_representation(spec, r, opts)
            dense_rt = sparse.dense_from_triplets(spec, *t_rt)
            assert_equal(dense, dense_rt)

            # Check triplets from dense returns original triplets:
            td = sparse.triplets_from_dense(dense_rt, bs)
            assert_equal(t_block.row_indices, td.row_indices)
            assert_equal(t_block.col_indices, td.col_indices)
            assert_equal(t_block.values, td.values)

    @pytest.mark.ipus(1)
    @pytest.mark.ipu_version("ipu2")
    def test_device_version_equality_ipu2(self):
        from ipu_sparse_ops import sparse
        bs = 16
        block_mask = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
        mask = np.kron(block_mask, np.ones(shape=[bs, bs])).astype(int)
        n_els = np.count_nonzero(mask)
        dense = np.zeros_like(mask)
        dense[np.nonzero(mask)] = np.arange(n_els)
        opts = {"metaInfoBucketOversizeProportion": 1}
        t = sparse.triplets_from_dense(dense)
        spec = sparse.matmul_spec_from_max(
            dense.shape[1], [2, dense.shape[0]], max_non_zeros=n_els, block_size=1, dtype=tf.float32)

        # from device
        device_r = sparse.representation_from_triplets(spec, *t, opts, ipu_version=0)
        device_t_rt = sparse.triplets_from_representation(spec, device_r, opts, ipu_version=0)

        # from version
        version_r = sparse.representation_from_triplets(spec, *t, opts, ipu_version=2)
        version_t_rt = sparse.triplets_from_representation(spec, version_r, opts, ipu_version=2)

        assert_equal(device_r.metainfo_state, version_r.metainfo_state)
        assert_equal(device_r.nz_values, version_r.nz_values)
        assert_equal(device_t_rt, version_t_rt)
