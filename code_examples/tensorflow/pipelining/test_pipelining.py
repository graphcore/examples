# Copyright 2020 Graphcore Ltd.

import inspect
import json
import os
import pytest
import subprocess
import unittest
from tempfile import TemporaryDirectory

import examples_tests.test_util as test_util


def run_pipelining_example(py_args):
    """Helper function to run the pipelining example"""
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    cmd = ['python', 'pipelining.py']
    args = [str(item) for sublist in py_args.items()
            for item in sublist if item != ""]
    cmd.extend(args)
    out = subprocess.check_output(cmd, cwd=cwd, universal_newlines=True)
    print(out)
    return out


class TestPipeliningTensorFlow(unittest.TestCase):
    """Tests for the pipelining TensorFlow code example"""

    @pytest.mark.category1
    @pytest.mark.ipus(2)
    def test_pipelining_convergence(self):
        """Run with default settings and check it converges"""
        out = run_pipelining_example({})
        # Get the final loss
        loss_regex = r"loss: ([\d.]+)"
        result = test_util.parse_results_with_regex(out, loss_regex)
        # Get the last loss
        loss = result[0][-1]
        self.assertGreater(loss, 0.001)
        self.assertLess(loss, 0.02)


    @pytest.mark.category1
    @pytest.mark.ipus(2)
    def test_pipelining_faster_than_sequential(self):

        def extract_cycles(report_dir):
            path = os.path.dirname(os.path.realpath(__file__))
            report_path = os.path.join(path, report_dir)
            exec_report_path = os.path.join(report_path,
                                            os.listdir(report_path)[0],
                                            'execution.json')
            with open(exec_report_path) as f:
                report = json.load(f)
                return report['simulation']['cycles']

        with TemporaryDirectory() as tmp_dir:
            """Run with grouped scheduling and profiling on,
            then run with sequential scheduling and profiling on;
            finally verify that the pipelined model runs in
            less cycles than the sequential model"""
            # Profile the pipelined model
            report_dir = os.path.join(tmp_dir, 'profile_dir1')
            out = run_pipelining_example({'--pipeline-schedule': 'Grouped',
                                          '--repeat-count': 1,
                                          '--synthetic-data': '',
                                          '--profile': '',
                                          '--json': '',
                                          '--report-directory': report_dir})
            # Fetch the total cycles from the execution report
            pipelining_cycles = extract_cycles(report_dir)
            self.assertGreater(pipelining_cycles, 0)

            # Profile the sequential model
            report_dir = os.path.join(tmp_dir, 'profile_dir2')
            out = run_pipelining_example({'--pipeline-schedule': 'Sequential',
                                          '--repeat-count': 1,
                                          '--synthetic-data': '',
                                          '--profile': '',
                                          '--json': '',
                                          '--report-directory': report_dir})
            # Fetch the total cycles from the execution report
            sequential_cycles = extract_cycles(report_dir)
            self.assertGreater(sequential_cycles, 0)

            # Assert pipelined model is faster
            self.assertLess(pipelining_cycles, sequential_cycles)



if __name__ == "__main__":
    unittest.main()
