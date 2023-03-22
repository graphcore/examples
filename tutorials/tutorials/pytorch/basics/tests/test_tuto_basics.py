# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).parent.parent


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_model():
    output = testing_util.run_command("python3 walkthrough.py", working_path, "Eval accuracy:")
    result_regex = r"Eval accuracy: ([\d.]+)\%"
    result_list = testing_util.parse_results_with_regex(output, result_regex)
    result = result_list[0][0]
    print(result)
    assert result >= 89


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_prediction():
    testing_util.run_command(
        "python3 walkthrough.py",
        working_path,
        ["IPU predicted class: Trouser", "CPU predicted class: Trouser"],
    )
