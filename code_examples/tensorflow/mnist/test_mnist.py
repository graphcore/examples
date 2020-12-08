# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import inspect
import os
import subprocess
import sys
import unittest
import pytest


def run_mnist():
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    print(cwd)
    cmd = ["python" + str(sys.version_info[0]), 'mnist_tf.py']
    out = subprocess.check_output(cmd, cwd=cwd).decode("utf-8")
    print("======")
    print(out)
    return out


class TestMnist(unittest.TestCase):
    """Test simple model on MNIST images on IPU. """

    @classmethod
    def setUpClass(cls):
        out = run_mnist()
        cls.final_acc = 0
        for line in out.split('\n'):
            if line.find('Test accuracy') != -1:
                cls.final_acc = float(line.split(":")[-1].strip())
                break

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_final_training_accuracy(self):
        self.assertGreater(self.final_acc, 0.9)
        self.assertLess(self.final_acc, 0.99)
