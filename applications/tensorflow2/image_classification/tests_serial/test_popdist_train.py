# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import unittest
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))
from test_common import run_train


class DistributedTraining(unittest.TestCase):

    def test_distributed_training(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        poprun_prefix = ['poprun', '--num-instances', '2', '--num-replicas', '4']
        output = run_train('--dataset-path', '/localdata/datasets/',
                           poprun_prefix=poprun_prefix)
        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)
