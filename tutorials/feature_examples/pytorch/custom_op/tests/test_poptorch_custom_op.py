# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path

import pytest
import torch
from tutorials_tests import testing_util
from filelock import FileLock

# Set seed to make test deterministic and we can test exact results
torch.manual_seed(42)

working_path = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def with_compiled_op():
    with FileLock(__file__ + ".lock"):
        testing_util.run_command(
            "make",
            working_path / "leaky_relu_example",
        )


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_program_run():
    # Check whether the model compiles and trains
    testing_util.run_command(
        "python3 poptorch_custom_op.py",
        working_path,
        "Epoch 4 | Loss: 0.67 | Accuracy: 74.23",
    )
