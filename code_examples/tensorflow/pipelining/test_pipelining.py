# Copyright 2020 Graphcore Ltd.

import inspect
import json
import os
import pytest
import subprocess
import unittest

import tests.test_util as test_util


def run_pipelining_example(py_args, gc_profile=False):
    """Helper function to run the pipelining example, optionally with gc-profile"""
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    cmd = []
    if gc_profile:
        cmd = ['gc-profile', '-d', 'profile_dir', '--']
    cmd.extend(['python', 'pipelining.py'])
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
        assert 0.001 <= loss <= 0.020

    @pytest.mark.category1
    @pytest.mark.ipus(2)
    def test_pipelining_faster_than_sequential(self):
        """Run with grouped scheduling and profiling on,
        then run with sequential scheduling and profiling on;
        finally verify that the pipelined model runs in
        less cycles than the sequential model"""
        # Profile the pipelined model
        out = run_pipelining_example({'--pipeline-schedule': 'Grouped',
                                      '--repeat-count': 1,
                                      '--synthetic-data': '',
                                      '--profile': ''},
                                     gc_profile=True)
        # Fetch the total cycles from the execution report
        path = os.path.dirname(os.path.realpath(__file__))
        exec_report_path = os.path.join(path, 'profile_dir/execution.json')
        with open(exec_report_path) as f:
            report = json.load(f)
            pipelining_cycles = report['simulation']['cycles']
        assert pipelining_cycles > 0

        # Profile the sequential model
        out = run_pipelining_example({'--pipeline-schedule': 'Sequential',
                                      '--repeat-count': 1,
                                      '--synthetic-data': '',
                                      '--profile': ''},
                                     gc_profile=True)
        # Fetch the total cycles from the execution report
        with open(exec_report_path) as f:
            report = json.load(f)
            sequential_cycles = report['simulation']['cycles']
        assert sequential_cycles > 0

        # Assert pipelined model is faster
        assert pipelining_cycles < sequential_cycles

        # Finally, delete the profiling directory
        subprocess.run(['rm', '-rf', 'profile_dir'], cwd=path)


if __name__ == "__main__":
    unittest.main()
