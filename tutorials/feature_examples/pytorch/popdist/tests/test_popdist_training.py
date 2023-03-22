# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import sys
import tempfile
from pathlib import Path

import torch
import pytest
from tutorials_tests import testing_util


@pytest.mark.category2
@pytest.mark.ipus(4)
def test_instances_in_sync_after_training():
    NUM_TOTAL_REPLICAS = 4
    NUM_INSTANCES = 2

    script_dir = Path(os.path.realpath(__file__)).parent.parent
    cmd = [
        "poprun",
        "--mpi-global-args",
        "--tag-output",
        "--num-replicas",
        str(NUM_TOTAL_REPLICAS),
        "--num-instances",
        str(NUM_INSTANCES),
        sys.executable,
        str(script_dir / "popdist_training.py"),
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        testing_util.run_command(cmd, cwd=temp_dir)

        checkpoint_0 = torch.load(
            os.path.join(temp_dir, "checkpoint-instance-0.pt"),
        )
        checkpoint_1 = torch.load(
            os.path.join(temp_dir, "checkpoint-instance-1.pt"),
        )

        # The final parameters should be the same for all instances.
        for k, v in checkpoint_0.items():
            if not (k.endswith("weight") or k.endswith("bias")):
                continue
            torch.testing.assert_allclose(v.data, checkpoint_1[k].data)
