# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import subprocess
import torch
import pytest
import poptorch
from torch.utils.data import dataset
from octconv_example import ClassificationModel, OctConvBlock
import tutorials_tests.testing_util as testing_util


def setup_function(*args, **kwargs):
    """
    Setup all tests in this module to use the same fixed random seed.
    """
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def dataset_folder(tmpdir_factory):
    return tmpdir_factory.mktemp("cifar10_data")


def run_script(script_name, parameters):
    cwd = os.path.dirname(os.path.abspath(__file__))
    param_list = parameters.split(" ")
    cmd = ["python3", script_name] + param_list
    cmd_str = " ".join(cmd)
    print(f"\nRunning:\n{cmd_str}\n")
    out = testing_util.run_command_fail_explicitly(cmd, cwd)

    return out


def get_test_accuracy(output):
    prefix = "Accuracy on test set:"
    pos_start = output.rfind(prefix)
    pos_end = pos_start + output[pos_start:].find("%")
    return float(output[pos_start + len(prefix) + 1 : pos_end])


@pytest.mark.parametrize("use_multi", [True, False])
def test_octconv_block(use_multi):
    block = OctConvBlock(3, 6, (0.0, 0.5), use_multi=use_multi)
    block.eval()  # Switch the model to inference mode

    # N, C, H, W
    x = torch.randn(5, 3, 10, 10)
    out = block(x)

    pop_block = poptorch.inferenceModel(block)
    pop_out = pop_block(x)

    for (cpu, ipu) in zip(out, pop_out):
        torch.testing.assert_allclose(cpu, ipu)


@pytest.mark.parametrize("conv_mode", ["vanilla", "octave", "multi-octave"])
def test_inference_model(conv_mode):
    model = ClassificationModel(conv_mode)
    model.eval()  # Switch the model to inference mode

    # N, C, H, W
    x = torch.randn(5, 3, 32, 32)
    out = model(x)

    pop_model = poptorch.inferenceModel(model)
    pop_out = pop_model(x)
    torch.testing.assert_allclose(out, pop_out)


@pytest.mark.parametrize("conv_mode", ["vanilla", "octave", "multi-octave"])
def test_training_model(conv_mode):
    model = ClassificationModel(conv_mode)

    # N, C, H, W
    x = torch.randn(5, 3, 32, 32)
    labels = torch.randint(low=1, high=10, size=(5,))
    out, loss = model(x, labels)

    model.train()  # Switch the model to training mode
    # Models are initialised in training mode by default, so the line above will
    # have no effect. Its purpose is to show how the mode can be set explicitly.

    pop_model = poptorch.trainingModel(model, poptorch.Options(), poptorch.optim.SGD(model.parameters(), lr=0.01))
    pop_out, pop_loss = pop_model(x, labels)
    torch.testing.assert_allclose(out, pop_out)
    torch.testing.assert_allclose(loss, pop_loss)


@pytest.mark.parametrize("conv_mode", ["vanilla", "octave", "multi-octave"])
def test_smoke(conv_mode, dataset_folder):
    out = run_script(
        "octconv_example.py",
        f"--conv-mode {conv_mode} --epochs 1 --data-dir {dataset_folder}",
    )

    assert get_test_accuracy(out) > 15.0


@pytest.mark.parametrize("conv_mode", ["vanilla", "octave", "multi-octave"])
def test_profiling(tmpdir, conv_mode):
    profile_dir = tmpdir.mkdir(f"profile_{conv_mode}")
    run_script("octconv_example.py", f"--conv-mode {conv_mode} --profile-dir {profile_dir}")
