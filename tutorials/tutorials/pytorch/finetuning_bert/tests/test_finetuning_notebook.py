# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest
import shutil
import tutorials_tests.testing_util as testing_util
import os

WORKING_PATH = Path(__file__).parent.parent.resolve()


@pytest.mark.category3
@pytest.mark.ipus(16)
@pytest.mark.parametrize("number_of_ipus", ["4", "16"])
def test_normal_operation(number_of_ipus):
    os.environ["NUM_AVAILABLE_IPU"] = number_of_ipus
    testing_util.run_command(
        "ipython Fine-tuning-BERT.py",
        WORKING_PATH,
        "Question: What speed-up can one expect from using sequence packing for" " training BERT on IPU?",
        timeout=5000,
    )

    shutil.rmtree(WORKING_PATH / "checkpoints", ignore_errors=True)
    shutil.rmtree(WORKING_PATH / "exe_cache", ignore_errors=True)
