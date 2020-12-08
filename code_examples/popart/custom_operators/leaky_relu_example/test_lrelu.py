# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import run_leaky_relu
from pathlib import Path


def setup_module(module):
    build_dir = Path(__file__).parent.resolve()
    subprocess.run("make clean", cwd=build_dir, shell=True, check=True)
    subprocess.run("make", cwd=build_dir, shell=True, check=True)


# Returns array of expected Leaky ReLU output given an input and alpha value
def get_expected(input_data, alpha):
    return [(x*alpha if x < 0 else x) for x in input_data]


@pytest.mark.ipus(1)
@pytest.mark.category1
# Tests command line script outputs successfully with various flag combinations
@pytest.mark.parametrize("input_data",
                         [[0.1],
                          [-0.1],
                          [-0.1, 0.1, 0.2, -1.2]])
@pytest.mark.parametrize("alpha",
                         [None,
                          0.01])
@pytest.mark.parametrize("run_on_ipu",
                         [True, False])
def test_leaky_relu_cmd_ln(input_data, alpha, run_on_ipu):
    py_version = "python" + str(sys.version_info[0])
    cmd = [py_version, "run_leaky_relu.py"]

    if alpha is not None:
        cmd.extend(["--alpha", str(alpha)])

    if run_on_ipu:
        cmd.append("--ipu")

    cmd.extend([str(i) for i in input_data])

    out = subprocess.check_output(cmd, cwd=os.path.dirname(__file__)).decode("utf-8")

    # Looks for output like: "{'LeakyRelu:0': array([....], dtype=float32)}""
    match = re.search(r"{'LeakyRelu:0': array\(\[[\-0-9. ,]*\], dtype=float32\)\}", out)

    assert(match)


@pytest.mark.ipus(1)
@pytest.mark.category1
# Tests that output of Leaky ReLU inference is correct with various combinations of inputs
@pytest.mark.parametrize("input_data",
                         [[0.1],
                          [0.1, 0.5],
                          [-0.1],
                          [-0.1, 0.1, 0.2, -1.2]])
@pytest.mark.parametrize("alpha",
                         [0.02,
                          0.01,
                          5.0])
@pytest.mark.parametrize("run_on_ipu",
                         [True, False])
def test_run_single_elmt(input_data, alpha, run_on_ipu):
    run_leaky_relu.load_custom_ops_lib()
    out = run_leaky_relu.build_and_run_graph(input_data, alpha, run_on_ipu)

    expected = get_expected(input_data, alpha)
    expected = np.array(expected).astype(np.float32)

    np.testing.assert_array_equal(list(out.values())[0],
                                  expected,
                                  "input:{}\nalpha:{}\nout:{}\nexpected:{}".format(input_data,
                                                                                   alpha,
                                                                                   out,
                                                                                   expected))
