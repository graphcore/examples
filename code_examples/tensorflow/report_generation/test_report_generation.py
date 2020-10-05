# Copyright 2020 Graphcore Ltd.
import os
import pytest
import unittest
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import run_python_script_helper, check_data_exists


def run_report_generation(**kwargs):
    """Helper function to run report generation python script"""
    out = run_python_script_helper(
        os.path.dirname(__file__), "report_generation_example.py", **kwargs
    )
    return out


class TestTensorflowReportGeneration(unittest.TestCase):
    """Simple tests for the report generation example"""

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_basic_usage(self):
        """Run with default arguments"""
        self._test_reports_helper({}, ["report.txt"])

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_execution_profiling(self):
        """Run with execution profiling on"""
        self._test_reports_helper({"--profile-execution": ""}, ["report.txt"])

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_json_report(self):
        """Run with option to generate json reports instead of text"""
        self._test_reports_helper({"--json-report": ""}, ["report.json"])

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_no_var_init(self):
        """Run with option to generate reports for the main graph only"""
        self._test_reports_helper({"--no-var-init-profiling": ""},
                                  ["report.txt"])

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_split_reports_no_execution(self):
        """Run with option to split compile and execution reports,
            but don't enable execution profiling"""
        self._test_reports_helper({"--split-reports": ""}, ["compile.txt"])

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_split_reports_with_execution(self):
        """Run with option to split compile and execution reports"""
        self._test_reports_helper({"--split-reports": "",
                                   "--profile-execution": ""},
                                  ["compile.txt", "execution.txt"])

    def _test_reports_helper(self, py_args, report_filenames):
        """Helper function for running the tests: it removes existing reports
           with the given names (if any), runs the script, checks that the
           reports have been generated and finally removes them."""
        report_path = os.path.dirname(os.path.realpath(__file__))
        # Remove existing reports
        for filename in report_filenames:
            self._remove_any_existing_report(report_path, filename)
        # Execute
        run_report_generation(**py_args)
        # Check expected reports exist
        for filename in report_filenames:
            self._check_file_exists_helper(report_path, filename)
        # Clean up
        for filename in report_filenames:
            self._remove_any_existing_report(report_path, filename)

    def _remove_any_existing_report(self, report_path, filename):
        """Helper function to check whether filename exists in report_path.
           If it does, the file is removed."""
        if check_data_exists(report_path, [filename]):
            os.remove(os.path.join(report_path, filename))

    def _check_file_exists_helper(self, report_path, filename):
        """Helper function to check whether report_generation.md exists
           in location report_path and raises an assertion error if it
           doesn't exist."""

        if not check_data_exists(report_path, [filename]):
            raise AssertionError(
                "{} does not exist in location {}".format(
                    filename, report_path
                )
            )
