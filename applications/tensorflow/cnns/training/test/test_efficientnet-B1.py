# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
Tests covering various CNN training options using the EfficientNet-B1 model.
"""

import unittest
import pytest

from test_common import get_csv, parse_csv, run_train, cifar10_data_dir


@pytest.mark.category2
@pytest.mark.ipus(4)
class TestEfficientNetB1Pipelining2IPUs(unittest.TestCase):
    """EfficientNet-B1 example over 4 IPUs.
    """

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--generated-data': '',
                           '--dataset': 'ImageNet',
                           '--model': 'efficientnet',
                           '--model-size': 'B1',
                           '--shards': 4,
                           '--pipeline': '',
                           '--gradient-accumulation-count': 128,
                           '--batch-size': 4,
                           '--no-validation': '',
                           '--xla-recompute': '',
                           '--available-memory-proportion': 0.2,
                           '--pipeline-schedule': 'Grouped',
                           '--iterations': 10,
                           '--pipeline-splits': 'block2a/c', 'block4a': 'block5c'})
        cls.out = out
        cls.training = get_csv(out, 'training.csv')

    def test_results(self):
        # test_iterations_completed
        self.assertEqual(self.training['iteration'][-1], 500)
        # test_number_of_parameters
        self.assertTrue('7794184' in self.out)
        # test_overall_batch_size
        self.assertTrue("Batch Size: 512" in self.out)
        # test image size
        self.assertTrue("240x240" in self.out)
