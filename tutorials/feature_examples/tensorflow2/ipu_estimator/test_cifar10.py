# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import pytest
from tutorials_tests.testing_util import parse_results_for_accuracy
import tutorials_tests.testing_util as testing_util


def run_cifar10(file_name):
    cwd = os.path.dirname(os.path.abspath(__file__))
    cmd = ["python3", f"cifar10_{file_name}.py"]
    out = testing_util.run_command_fail_explicitly(cmd, cwd)

    parse_results_for_accuracy(out, [81.0], 6.0)
    return out


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_ipuestimator():
    run_cifar10("ipu_estimator")


@pytest.mark.category2
@pytest.mark.ipus(2)
def test_replica():
    run_cifar10("ipu_estimator_replica")


@pytest.mark.category2
@pytest.mark.ipus(2)
def test_pipeline():
    run_cifar10("ipu_pipeline_estimator")
