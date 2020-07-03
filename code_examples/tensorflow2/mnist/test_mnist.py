# Copyright 2020 Graphcore Ltd.
import unittest
import os
import pytest
import tensorflow as tf
from tests.test_util import SubProcessChecker

working_path = os.path.dirname(__file__)


@pytest.mark.category1
@pytest.mark.ipus(1)
class TensorFlow2Mnist(SubProcessChecker):
    """Integration tests for TensorFlow 2 MNIST example"""

    @unittest.skipIf(tf.__version__[0] != '2', "Needs TensorFlow 2")
    def test_default_commandline(self):
        self.run_command("python3 mnist.py",
                         working_path,
                         "Epoch 2/")
