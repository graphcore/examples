# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import unittest
import tensorflow as tf

from scripts.dataset_benchmark import estimate_ds_throughput


class FunctionalCheckOfDatasetBenchmarking(unittest.TestCase):
    def test_benchmark_output(self):
        batch_size = 8
        ds_size = 10000
        images: tf.Tensor = tf.random.truncated_normal(
            [32, 32, 3], dtype=tf.float32, mean=127, stddev=60, name="inputs"
        )
        labels = tf.ones([])
        ds = tf.data.Dataset.from_tensors((images, labels)).repeat(ds_size)

        throughput = estimate_ds_throughput(ds, ds_size, epochs=2, micro_batch_size=batch_size, num_instances=1)
        assert isinstance(throughput, float)
        assert throughput > 0
