# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pathlib
import sys
import pytest
import tutorials_tests.testing_util as testing_util


working_directory = pathlib.Path(__file__).absolute().parent.parent


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_no_parameters():
    """Run with default settings and check that a loss is given."""

    out = testing_util.run_command_fail_explicitly([sys.executable, "recompute_lstm.py"], working_directory)
    assert "loss:" in out, "Model didn't compute a loss."


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_parameters():
    """Run with different settings and check that a loss and checkpoint warning is given."""

    out = testing_util.run_command_fail_explicitly(
        [sys.executable, "recompute_lstm.py", "--seq-len", "512", "--checkpoints", "8"],
        working_directory,
    )
    assert "loss:" in out, "Model didn't compute a loss."
    assert "can't be evenly divided" in out, "Warning over function outlining not given."
