# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import pathlib
import pytest
import tensorflow as tf
import tutorials_tests.testing_util as testing_util

working_path = pathlib.Path(__file__).parents[1]

"""Integration tests for TensorFlow 2 MNIST example"""


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_default_commandline():
    testing_util.run_command("python3 mnist_code_only.py", working_path, "Epoch 2/")
