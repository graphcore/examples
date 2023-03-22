# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import logging
import os
import re
import sys
from pathlib import Path

import pytest

from tutorials_tests import testing_util

script_dir = Path(os.path.abspath(__file__)).parents[1]
accuracy_tolerance = 0.03


@pytest.mark.category1
@pytest.mark.ipus(4)
def test_training_then_inference(tmpdir):
    run_script_and_check_loss(
        num_instances=2,
        num_total_replicas=4,
        script="popdist_training.py",
        expected_acc=0.7588,
        working_dir=tmpdir,
    )
    run_script_and_check_loss(
        num_instances=4,
        num_total_replicas=4,
        script="popdist_inference.py",
        expected_acc=0.7731,
        working_dir=tmpdir,
    )


def test_training_1_replica_per_instance(tmpdir):
    run_script_and_check_loss(
        num_instances=2,
        num_total_replicas=2,
        script="popdist_training.py",
        expected_acc=0.8228,
        working_dir=tmpdir,
    )


def run_script_and_check_loss(num_instances, num_total_replicas, script, expected_acc, working_dir):
    cmd = [
        "poprun",
        "--num-replicas",
        str(num_total_replicas),
        "--num-instances",
        str(num_instances),
        sys.executable,
        str(script_dir / script),
    ]

    logging.info(f"Executing: {cmd} in {working_dir}")
    out = testing_util.run_command_fail_explicitly(cmd, cwd=working_dir)
    acc = float(re.findall(r"accuracy: \d+\.\d+", out)[-1].split(" ")[-1])
    assert abs(expected_acc - acc) < accuracy_tolerance, (
        f"Measured accuracy {acc} does fall within the "
        f"{accuracy_tolerance} tolerance around the expected "
        f"accuracy of {expected_acc}"
    )
