# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from tutorials_tests import testing_util

working_path = Path(__file__).parent.parent


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_model_data_fp16():
    # Check whether the model compiles and executes in FP16
    testing_util.run_command(
        "python3 walkthrough.py --execution-half",
        working_path,
        "Eval accuracy on IPU",
    )


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_model_data_accum_fp16():
    # Check whether the model compiles and executes in FP16
    testing_util.run_command(
        "python3 walkthrough.py --execution-half --optimizer-half",
        working_path,
        "Eval accuracy on IPU",
    )


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_stochastic_rounding():
    # Check whether the model compiles and executes in FP16
    testing_util.run_command(
        "python3 walkthrough.py --execution-half --optimizer-half --stochastic-rounding",
        working_path,
        "Eval accuracy on IPU",
    )


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_all_fp16():
    # Check whether the model compiles and executes in FP16
    testing_util.run_command(
        "python3 walkthrough.py --execution-half --optimizer-half --stochastic-rounding --partials-half",
        working_path,
        "Eval accuracy on IPU",
    )
