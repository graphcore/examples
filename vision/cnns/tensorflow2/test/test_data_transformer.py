# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import unittest
from pathlib import Path
import sys
import tensorflow as tf
import numpy as np

sys.path.append(str(Path(__file__).absolute().parent.parent))

from datasets.data_transformer import DataTransformer
from custom_exceptions import DimensionError, UnsupportedFormat


class CheckNormalise(unittest.TestCase):
    def test_unsuported_dimension(self):
        ds = tf.data.Dataset.from_tensor_slices([1, 2, 3])
        with self.assertRaises(DimensionError):
            _ = DataTransformer.normalization(ds)

    def test_unsuported_format(self):
        ds = [1, 2, 3]
        with self.assertRaises(UnsupportedFormat):
            _ = DataTransformer.normalization(ds)

    def test_normalisation_values_and_type_fp32(self):
        ds = tf.data.Dataset.from_tensors(([1, 1, 1], 4))
        ds = DataTransformer.normalization(ds, scale=1/3.)
        for f, l in ds:
            np.testing.assert_almost_equal(
                list(f.numpy()), [1 / 3., 1 / 3., 1 / 3.], 5)
            assert(f[0].dtype == tf.float32)
            assert(l.dtype == tf.int32)

    def test_normalisation_values_and_type_fp16(self):
        ds = tf.data.Dataset.from_tensors(([1, 1, 1], 4))
        ds = DataTransformer.normalization(ds, img_type=tf.float16)
        for f, l in ds:
            np.testing.assert_almost_equal(
                list(f.numpy()), [1 / 255., 1 / 255., 1 / 255.], 5)
            assert(f[0].dtype == tf.float16)
            assert(l.dtype == tf.int32)
