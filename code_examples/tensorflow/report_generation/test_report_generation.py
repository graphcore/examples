#!/usr/bin/python
# Copyright 2019 Graphcore Ltd.

import inspect
import os
import unittest

import tests.test_util as test_util


def run_report_generation():
    """Helper function to run report generation python script"""
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    out = test_util.run_python_script_helper(cwd, "report_generation_example.py")
    return out


class TestTensorflowReportGeneration(unittest.TestCase):
    """Simple test for the report generation example"""

    def test_report_generation(self):
        """Simple test runs the report generation python script and
           verifies it has produced the correct file"""
        run_report_generation()
        self._check_file_exists_helper(
            os.path.dirname(os.path.realpath(__file__))
        )

    def _check_file_exists_helper(self, path_to_file):
        """Helper function to check whether report_generation.md exists
           in location path_to_file and raises an assertion error if it
           doesn't exist"""

        files_to_generate = ["report.txt"]

        if not test_util.check_data_exists(path_to_file, files_to_generate):
            raise AssertionError(
                "report.txt does not exist in location {}".format(
                    path_to_file
                )
            )


if __name__ == "__main__":
    unittest.main()
