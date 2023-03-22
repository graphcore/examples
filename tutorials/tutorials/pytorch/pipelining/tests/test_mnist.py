# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import inspect
import os
from tutorials_tests import testing_util
import torch
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from mnist_pipeline import accuracy

torch.manual_seed(0)


def run_poptorch_mnist(parameters):
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    param_list = parameters.split(" ")
    cmd = ["python3", "../mnist_pipeline.py"] + param_list
    cmd_str = " ".join(cmd)
    print(f"\nRunning:\n{cmd_str}\n")
    print(cmd)
    out = testing_util.run_command_fail_explicitly(cmd, cwd=cwd)
    return out


@pytest.mark.category2
def test_accuracy_calculation():
    """Training tests for MNIST in PopTorch"""
    pred = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.5, 0.4], [0.6, 0.01, 0.49], [0.09, 0.11, 0.8]])
    label = torch.tensor([0, 1, 2, 2])
    acc = accuracy(pred, label)
    assert acc == 75


@pytest.mark.category2
@pytest.mark.ipus(4)
@pytest.mark.parametrize("strategy", ["sharded", "pipelined"])
@pytest.mark.parametrize("offload", [True, False])
def test_test_final_training_accuracy(strategy, offload):
    if offload:
        out = run_poptorch_mnist(f"--strategy {strategy} --epochs 2 " f"--offload-optimiser")
    else:
        out = run_poptorch_mnist(f"--strategy {strategy} --epochs 2")
    final_acc = 0.0
    for line in out.split("\n"):
        if line.find("Accuracy on test set:") != -1:
            final_acc = float(line.split(": ")[-1].strip()[:-1])
            break
    assert final_acc > 97.5
    assert final_acc < 99.9
