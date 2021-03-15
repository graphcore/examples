# Copyright (c) 2020 Graphcore Ltd. All rights reserved.


import os
from pathlib import Path
import pytest
import tensorflow as tf
import unittest

from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).resolve().parent.parent


@pytest.mark.category2
@unittest.skipIf(tf.__version__[0] != '2', "Needs TensorFlow 2")
class TestTensorFlow2InspectingTensors(SubProcessChecker):
    """Integration tests for TensorFlow 2 Inspecting Tensors example"""


    @pytest.mark.ipus(2)
    def test_default_commandline(self):
        """ Test the default command line which selects the PipelineSequential model
        """
        self.run_command("python3 mnist.py",
                         working_path,
                         [r"Gradients callback\n" +
                          r"key: Dense_128\/bias:0_grad shape: \(2000, 128\)",
                          r"Multi-layer activations callback\n" +
                          r"key: Dense_128_acts shape: \(8000, 32, 128\)\n" +
                          r"key: Dense_10_acts shape: \(8000, 32, 10\)"])


    @pytest.mark.ipus(1)
    def test_model(self):
        """ Test the Model
        """
        self.run_command("python3 mnist.py --model-type Model --epochs 1 --steps-per-epoch 500",
                         working_path,
                         [r"Gradients callback\n" +
                          r"key: Dense_128\/bias:0_grad shape: \(500, 128\)",
                          r"Single layer activations callback\n" +
                          r"key: Dense_128_acts shape: \(500, 32, 128\)"])


    @pytest.mark.ipus(1)
    def test_model_gradient_accumulation(self):
        """ Test the Model with gradient accumulation enabled
        """
        self.run_command("python3 mnist.py --model-type Model --epochs 1"
                         " --steps-per-epoch 500"
                         " --gradient-accumulation",
                         working_path,
                         [r"Gradients callback\n" +
                          r"key: Dense_128\/bias:0_grad shape: \(500, 128\)",
                          r"Single layer activations callback\n" +
                          r"key: Dense_128_acts shape: \(2000, 32, 128\)"])


    @pytest.mark.ipus(1)
    def test_model_gradient_accumulation_pre_accumulated_gradients(self):
        """ Test the Model, outfeeding the pre-accumulated gradients
        """
        self.run_command("python3 mnist.py --model-type Model --epochs 1"
                         " --steps-per-epoch 500 --outfeed-pre-accumulated-gradients"
                         " --gradient-accumulation",
                         working_path,
                         [r"Gradients callback\n" +
                          r"key: Dense_128\/bias:0_grad shape: \(2000, 128\)",
                          r"Single layer activations callback\n" +
                          r"key: Dense_128_acts shape: \(2000, 32, 128\)"])


    @pytest.mark.ipus(2)
    def test_pipeline_model(self):
        """ Test the PipelineModel, outfeeding the pre-accumulated gradients
        """
        self.run_command("python3 mnist.py --model-type PipelineModel --epochs 1"
                         " --steps-per-epoch 500 --outfeed-pre-accumulated-gradients"
                         " --activations-filters Dense_10",
                         working_path,
                         [r"Gradients callback\n" +
                          r"key: Dense_128\/bias:0_grad shape: \(2000, 128\)",
                          r"Multi-layer activations callback\n" +
                          r"key: Dense_10_acts shape: \(2000, 32, 10\)"])


    @pytest.mark.ipus(1)
    def test_sequential(self):
        """ Test the Sequential model
        """
        self.run_command("python3 mnist.py --model-type Sequential --epochs 1"
                         " --steps-per-epoch 500 --gradients-filters none",
                         working_path,
                         [r"Gradients callback\n" +
                          r"key: Dense_10\/bias:0_grad shape: \(500, 10\)",
                          r"key: Dense_128\/bias:0_grad shape: \(500, 128\)",
                          r"Single layer activations callback\n" +
                          r"key: Dense_128_acts shape: \(500, 32, 128\)"])
