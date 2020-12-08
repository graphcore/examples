# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import unittest
import os
import pytest
import tensorflow as tf
from examples_tests.test_util import SubProcessChecker

working_path = os.path.dirname(__file__)


class TensorFlow2Imdb(SubProcessChecker):
    """Integration tests for TensorFlow 2 IMDB example"""
    @pytest.mark.category2
    @pytest.mark.ipus(2)
    @unittest.skipIf(tf.__version__[0] != '2', "Needs TensorFlow 2")
    def test_pipeline(self):
        self.run_command("python imdb.py",
                         working_path,
                         "Epoch 2/")

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    @unittest.skipIf(tf.__version__[0] != '2', "Needs TensorFlow 2")
    def test_sequential_pipeline(self):
        self.run_command("python imdb_sequential.py",
                         working_path,
                         "Epoch 2/")

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    @unittest.skipIf(tf.__version__[0] != '2', "Needs TensorFlow 2")
    def test_single_ipu(self):
        self.run_command("python imdb_single_ipu.py",
                         working_path,
                         "Epoch 3/")

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    @unittest.skipIf(tf.__version__[0] != '2', "Needs TensorFlow 2")
    def test_single_ipu_sequential(self):
        self.run_command("python imdb_single_ipu_sequential.py",
                         working_path,
                         "Epoch 3/")
