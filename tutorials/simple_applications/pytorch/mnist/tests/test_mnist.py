# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import sys
import pytest
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from metrics import accuracy
import tutorials_tests.testing_util as testing_util


def run_poptorch_mnist(**kwargs):
    cwd = Path(__file__).parent.parent
    cmd = ["python3", "mnist_poptorch.py"]
    out = testing_util.run_command_fail_explicitly(cmd, cwd)

    return out


def test_accuracy_calculation():
    pred = torch.tensor(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.5, 0.4],
            [0.6, 0.01, 0.49],
            [0.09, 0.11, 0.8],
        ]
    )
    label = torch.tensor([0, 1, 2, 2])
    acc = accuracy(pred, label)
    assert acc == 75


@pytest.mark.ipus(1)
def test_test_final_training_accuracy():
    out = run_poptorch_mnist()
    final_acc = 0.0
    for line in out.split("\n"):
        if line.find("Accuracy on test set:") != -1:
            final_acc = float(line.split(": ")[-1].strip()[:-1])
            break

    assert final_acc > 89
    assert final_acc < 99.9
