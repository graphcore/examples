# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import re
import subprocess
import unittest

import numpy as np
import pytest
# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import run_python_script_helper, check_data_exists


class TestTensorflowCustomOp(unittest.TestCase):
    """Simple tests for the TensorFlow custom op example"""

    @classmethod
    def setUpClass(cls):
        cls.path = os.path.dirname(__file__)
        compile_custom_op(cls.path)

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_tf_code(self):
        """Run the python script and check the result"""
        # Run the script and capture the output
        out = run_python_script_helper(self.path, "tf_code.py")
        # Get the first and the second line of output
        ipu_res, target_res = out.split("\n")[:-1]
        # Convert these lines to arrays, in turn
        list_regex = r"\[.*\]$"
        match = re.search(list_regex, ipu_res)
        string_vals = match.group()[1:-1].split()
        ipu_arr = np.array([float(val)
                            for val in string_vals], dtype=np.float32)
        match = re.search(list_regex, target_res)
        string_vals = match.group()[1:-1].split()
        target_arr = np.array([float(val)
                               for val in string_vals], dtype=np.float32)
        # Finally, check that the results are reasonably close
        assert np.allclose(ipu_arr, target_arr), (
            "Output value {} does not "
            "equal expected value {}".format(ipu_arr, target_arr)
        )
        # Clean up
        subprocess.run(["make", "clean"], cwd=self.path)


def compile_custom_op(path):
    """Runs the make command to build the custom op objects"""
    files_to_generate = ["custom_codelet.gp", "libcustom_op.so"]

    if check_data_exists(path, files_to_generate):
        print("Objects already present, cleaning...")
        subprocess.run(["make", "clean"], cwd=path)

    completed = subprocess.run("make", cwd=path)
    if completed.returncode != 0 or not check_data_exists(path, files_to_generate):
        raise Exception("Custom op compilation failed")

    print("Successfully compiled custom op")
