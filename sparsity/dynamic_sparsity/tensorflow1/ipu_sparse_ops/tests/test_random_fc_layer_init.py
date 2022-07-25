# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import os
import sys
import numpy as np
import pytest
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cwd, '..', '..'))


def assert_not_equal(a, b):
    assert not np.array_equal(a, b)


@pytest.mark.usefixtures("ipu_sparse_ops")
class TestBuildAndRun(SubProcessChecker):

    def test_random_fc_layers(self):
        from ipu_sparse_ops import layers
        random_seed = 101
        random_gen = np.random.default_rng(seed=random_seed)
        fc1 = layers.SparseFcLayer.from_random_generator(
            64, [128, 256], 0.1,
            block_size=1,
            values_initialiser_gen=random_gen.standard_normal,
            indices_initialiser_gen=random_gen,
            matmul_options={"metaInfoBucketOversizeProportion": 0.1},
            name='sparse_fc_from_random',
            use_bias=False, relu=False)
        fc2 = layers.SparseFcLayer.from_random_generator(
            64, [128, 256], 0.1,
            block_size=1,
            values_initialiser_gen=random_gen.standard_normal,
            indices_initialiser_gen=random_gen,
            matmul_options={"metaInfoBucketOversizeProportion": 0.1},
            name='sparse_fc_from_random',
            use_bias=False, relu=False)
        t1 = fc1.get_triplets()
        t2 = fc2.get_triplets()
        for a, b in zip(t1, t2):
            assert_not_equal(a, b)
