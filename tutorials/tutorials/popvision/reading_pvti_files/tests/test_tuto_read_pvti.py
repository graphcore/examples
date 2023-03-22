# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from tutorials_tests import testing_util

working_path = Path(__file__).parent.parent


@pytest.mark.category2
def test_tutorial_code():
    expected_strings = [
        "Number of processes",
        "Number of threads on process",
        "Number of events on thread",
        "The longest epoch",
        "The shortest epoch",
        "Epochs took",
    ]
    testing_util.run_command("python3 walkthrough.py", working_path, expected_strings)
