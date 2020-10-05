# Copyright 2020 Graphcore Ltd.
import numpy as np
import os
import unittest
import pytest
from more_itertools import locate

# NOTE: The imports below are dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import run_python_script_helper
from examples_tests.test_util import assert_result_equals_tensor_value
from examples_tests.test_util import assert_result_equals_string

# Strings used to identify lines from trace that
# indicate graph compilation and device attachment respectively.
# Using trace here rather than events because it is less intrusive
# and avoids polluting the example itself with unnecessary complexity.
COMPILE_STRING = "Compiled "
ATTACH_STRING = "attached to "


def parse_output(out):
    """Helper to parse output (stdout/stderr) and return a
       dictionary that includes the result plus line indices
       for compilation and attachments."""
    lines = out.splitlines()
    compile_list = list(locate(lines, lambda l: COMPILE_STRING in l))
    attach_list = list(locate(lines, lambda l: ATTACH_STRING in l))
    return {'result': lines[-1], 'compile': compile_list, 'attach': attach_list}


def run_connection_type(connection_type):
    """Helper to run connect_type.py with specific connection type,
       capture the output, and parse the result."""
    kwargs = {"--connection_type": connection_type}
    out = run_python_script_helper(os.path.dirname(__file__),
                                   "connection_type.py",
                                   want_std_err=True,
                                   **kwargs)
    result = parse_output(out)
    print("result {}".format(result))
    return result


# Set of tests.
class TestTensorFlowConnectionType(unittest.TestCase):
    """High-level integration tests for tensorflow connection type examples"""


    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_connection_type_always(self):
        """Connection type ALWAYS"""
        result = run_connection_type("ALWAYS")
        # Assert correct result.
        assert_result_equals_tensor_value(
            result['result'], np.array([3.0, 8.0], dtype=np.float32)
        )
        # Assert single occurences of attach and compile
        # with attach occuring first.
        assert(len(result['attach']) == 1), "Missing attach"
        assert(len(result['compile']) == 1), "Missing compile"
        assert(result['attach'][0] < result['compile'][0]), "Compile before attach"


    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_connection_type_on_demand(self):
        """Connection type ON_DEMAND"""
        result = run_connection_type("ON_DEMAND")
        # Assert correct result.
        assert_result_equals_tensor_value(
            result['result'], np.array([3.0, 8.0], dtype=np.float32)
        )
        # Assert single occurences of attach and compile
        # with compilation occuring first.
        assert(len(result['attach']) == 1), "Missing attach"
        assert(len(result['compile']) == 1), "Missing compile"
        assert(result['attach'][0] > result['compile'][0]), "Compile after attach"


    @pytest.mark.category1
    def test_connection_type_never(self):
        """Connection type NEVER"""
        result = run_connection_type("NEVER")
        # Assert correct result.
        assert_result_equals_string(result['result'], "Compiled")
        # Assert single occurence of compile without attach.
        assert(len(result['attach']) == 0), "Unexpected attach"
        assert(len(result['compile']) == 1), "Missing compile"


if __name__ == "__main__":
    unittest.main()
