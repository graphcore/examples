# Copyright (c) 2020 Graphcore Ltd. All rights reserved.


from pathlib import Path
import pytest
import tensorflow as tf

from tutorials_tests.testing_util import run_command

working_path = Path(__file__).resolve().parent.parent


@pytest.mark.category2
@pytest.mark.skipif(tf.__version__[0] != "2", reason="Needs TensorFlow 2")
class TestTensorFlow2InspectingTensors:
    """Integration tests for TensorFlow 2 Inspecting Tensors example"""

    @pytest.mark.ipus(2)
    def test_default_commandline(self):
        """Test the default command line which selects the pipelined Sequential model"""
        run_command(
            "python3 mnist.py",
            working_path,
            [
                r"Gradients callback\n" + r"key: Dense_128\/bias:0_grad shape: \(500, 128\)",
                r"Multi-layer activations callback\n"
                + r"key: Dense_128_acts shape: \(2000, 32, 128\)\n"
                + r"key: Dense_10_acts shape: \(2000, 32, 10\)",
            ],
        )

    @pytest.mark.ipus(1)
    def test_model(self):
        """Test the Model"""
        run_command(
            "python3 mnist.py --model-type Model --no-pipelining --epochs 1 --steps-per-epoch 500",
            working_path,
            [
                r"Gradients callback\n" + r"key: Dense_128\/bias:0_grad shape: \(500, 128\)",
                r"Single layer activations callback\n" + r"key: Dense_128_acts shape: \(500, 32, 128\)",
            ],
        )

    @pytest.mark.ipus(1)
    def test_model_gradient_accumulation(self):
        """Test the Model with gradient accumulation enabled"""
        run_command(
            "python3 mnist.py --model-type Model --no-pipelining"
            " --epochs 1 --steps-per-epoch 500"
            " --use-gradient-accumulation",
            working_path,
            [
                r"Gradients callback\n" + r"key: Dense_128\/bias:0_grad shape: \(125, 128\)",
                r"Single layer activations callback\n" + r"key: Dense_128_acts shape: \(500, 32, 128\)",
            ],
        )

    @pytest.mark.ipus(1)
    def test_model_gradient_accumulation_pre_accumulated_gradients(self):
        """Test the Model, outfeeding the pre-accumulated gradients"""
        run_command(
            "python3 mnist.py --model-type Model --no-pipelining"
            " --epochs 1 --steps-per-epoch 500 "
            " --outfeed-pre-accumulated-gradients --use-gradient-accumulation",
            working_path,
            [
                r"Gradients callback\n" + r"key: Dense_128\/bias:0_grad shape: \(500, 128\)",
                r"Single layer activations callback\n" + r"key: Dense_128_acts shape: \(500, 32, 128\)",
            ],
        )

    @pytest.mark.ipus(2)
    def test_pipeline_model(self):
        """Test the pipelined Model, outfeeding the pre-accumulated gradients"""
        run_command(
            "python3 mnist.py --model-type Model --epochs 1"
            " --steps-per-epoch 500 --outfeed-pre-accumulated-gradients"
            " --activations-filters Dense_10",
            working_path,
            [
                r"Gradients callback\n" + r"key: Dense_128\/bias:0_grad shape: \(500, 128\)",
                r"Multi-layer activations callback\n" + r"key: Dense_10_acts shape: \(500, 32, 10\)",
            ],
        )

    @pytest.mark.ipus(1)
    def test_sequential(self):
        """Test the Sequential model"""
        run_command(
            "python3 mnist.py --model-type Sequential --no-pipelining"
            " --epochs 1 --steps-per-epoch 500"
            " --gradients-filters none",
            working_path,
            [
                r"Gradients callback\n" + r"key: Dense_10\/bias:0_grad shape: \(500, 10\)",
                r"key: Dense_128\/bias:0_grad shape: \(500, 128\)",
                r"Multi-layer activations callback\n" + r"key: Dense_128_acts shape: \(500, 32, 128\)",
            ],
        )
