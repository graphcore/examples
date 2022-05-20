# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
Tests covering various CNN training options using the EfficientNet-B1 model.
"""

import unittest
import pytest
import sys
from pathlib import Path
from examples_tests.test_util import SubProcessChecker

sys.path.append(str(Path(__file__).absolute().parent.parent))

from test_common import get_csv, run_train


@pytest.mark.ipus(4)
@pytest.mark.ipu_version("ipu2")
class TestEfficientNetB1Pipelining2IPUs(SubProcessChecker):
    """EfficientNet-B1 example over 4 IPUs.
    """

    def setUp(self):
        out = run_train(
            self,
            **{
                '--generated-data': '',
                '--dataset': 'ImageNet',
                '--model': 'efficientnet',
                '--model-size': 'B1',
                '--shards': 4,
                '--pipeline': '',
                '--gradient-accumulation-count': 128,
                '--micro-batch-size': 4,
                '--no-validation': '',
                '--enable-recomputation': '',
                '--available-memory-proportion': 0.2,
                '--pipeline-schedule': 'Grouped',
                '--iterations': 10,
                '--pipeline-splits': 'block2a/c', 'block4a': 'block5c',
                '--fused-preprocessing': ''})
        self.out = out
        self.training = get_csv(out, 'training.csv')

    def test_results(self):
        # test_iterations_completed
        self.assertEqual(self.training['iteration'][-1], 500)
        # test_number_of_parameters
        self.assertTrue('7794472' in self.out)
        # test_overall_batch_size
        self.assertTrue("Batch Size: 512" in self.out)
        # test image size
        self.assertTrue("240x240" in self.out)
