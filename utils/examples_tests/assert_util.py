# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import numpy as np
import re


def assert_result_equals_tensor_value(output, tensor):
    """Searches for a single tensor result in the first line of the output


    Searches the first line of the string output for a line with format
    '[array([3., 8.], dtype=float32)]' and asserts its equal to the numpy
    tensor argument

    Args:
        output: String containing the string representation of a numpy
            tensor
        tensor: numpy tensor representing the expected result

    Returns:
        None

    Raises:
        Assertion Error: Output is not in correct format
        Assertion Error: Output does not contain a string representation
            of a numpy array
        Assertion Error: Output numpy array does not equal the expected
            numpy array
    """
    # TODO - np representation over multiple lines
    # TODO - large np array output
    # TODO - multiple dimension np output
    list_regex = r"^\[.*?\]$"
    np_array_str_regex = r"array\(.*?, dtype=.*?\)$"
    first_line = output.split("\n")[0]
    if not re.match(list_regex, first_line):
        raise AssertionError(
            "Result not in expected string format."
            "  Expecting stringified list "
            " eg. [array([3., 8.], dtype=float32)]"
        )

    contents = first_line[1:-1]
    if not re.match(np_array_str_regex, contents):
        raise AssertionError(
            "Expecting numpy representation "
            "array with dtype "
            "eg. array([3., 8.], dtype=float32)"
        )

    assert contents == np.array_repr(tensor), (
        "Output value {} does not "
        "equal expected value {}".format(np.array_repr(contents), tensor)
    )


def assert_result_equals_string(output, expected):
    """Checks output line equals expected string

    Args:
        output: String representing the output of a test.
        expected: String of expected result.

    Returns:
        None

    Raises:
        Assertion Error: Output string does not equal the expected
            string
    """

    assert output == expected, (
        "Output string {} does not "
        "equal expected string {}".format(output, expected)
    )
