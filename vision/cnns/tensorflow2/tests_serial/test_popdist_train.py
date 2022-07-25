# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))
from test_common import run_train
import os
import shutil
from examples_tests.test_util import SubProcessChecker


class DistributedTraining(SubProcessChecker):

    def test_distributed_training(self):

        poprun_prefix = ['poprun',
                         '--only-output-from-instance', '0',
                         '--num-instances', '2',
                         '--num-replicas', '2']
        output = run_train(self, '--num-epochs', '5', poprun_prefix=poprun_prefix)

        shutil.rmtree('cifar10')
        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)
