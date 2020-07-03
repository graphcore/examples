# Copyright 2020 Graphcore Ltd.
import os
import re
import sys
import unittest
import pytest
from statistics import mean

import pexpect

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from tests.test_util import (
    run_python_script_helper,
    run_test_helper,
    get_minimum_with_tolerance,
    get_maximum_with_tolerance,
    check_data_exists,
    parse_results_with_regex,
)


def run_tensorflow_nmt(**kwargs):
    """Helper function to run nmt tensorflow python script with
       command line arguments"""
    out = run_python_script_helper(os.path.dirname(__file__), "nmt-tf.py", **kwargs)
    return out


class TestTensorflowNmtSequenceModelling(unittest.TestCase):
    """High-level integration tests for NMT model in tensorflow in training
       and inference"""

    @classmethod
    def setUpClass(cls):
        file_path = os.path.dirname(__file__)
        cls.generic_arguments = {
            "--attention": "luong",
            "--batch-size": 1,
            "--embedding-size": 32,
            "--num-units": 512,
            "--num-layers": 1,
            "--seed": 1984,
            "--sequence-length": 20,
            "--batches-per-step": 100,
            "--iterations": 10,
        }

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_nmt_tf_bi_false_luong(self):
        """Trains a model with bidirectional encoder with luong attention and
           performs inference on weights after 1000 steps"""
        py_args = self.generic_arguments.copy()
        train_and_infer_helper(
            py_args,
            last_measured_loss=14.1
        )

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_nmt_tf_bi_false_luong_host_embeddings(self):
        """Trains a model with bidirectional encoder with luong attention and
           performs inference on weights after 1000 steps"""
        py_args = self.generic_arguments.copy()
        py_args["--host-embeddings"] = ""
        train_and_infer_helper(
            py_args,
            last_measured_loss=18.4,
            infer=False
        )

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_nmt_tf_bi_true_luong(self):
        """Trains a model with directional encoder with luong attention and
           performs inference on weights after 1000 steps"""
        py_args = self.generic_arguments.copy()
        py_args["--bi"] = ""
        train_and_infer_helper(
            py_args,
            last_measured_loss=20.3
        )

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_nmt_tf_bi_true_bahdanau(self):
        """Trains a model with bidirectional encoder with bahdanau attention and
           performs inference on weights after 1000 steps"""
        py_args = self.generic_arguments.copy()
        py_args["--attention"] = "bahdanau"
        train_and_infer_helper(
            py_args,
            last_measured_loss=24.0
        )

    @pytest.mark.ipus(1)
    @pytest.mark.category2
    def test_nmt_save_graph_checkpoint_interact_args(self):
        """Exercises the save graph and checkpoint command line arguments.
           These scripts must be run in this order."""
        self._arg_test_helper("--save-graph")
        self._arg_test_helper("--ckpt")
        self._interaction_argument_test_helper()

    def _arg_test_helper(self, *args):
        """Helper function that exercises the command line arguments in the
           python model"""
        py_args = self.generic_arguments.copy()
        for arg in args:
            py_args[arg] = ""
        run_test_helper(run_tensorflow_nmt, **py_args)

    def _interaction_argument_test_helper(self):
        """Helper function that starts the model in interactive mode and
           inputs a value"""
        py_version = "python{}".format(sys.version_info[0])
        p = pexpect.spawn(
            "{} nmt-tf.py --interact".format(py_version), cwd=os.path.dirname(__file__)
        )
        p.logfile = sys.stdout.buffer
        p.expect("Enter a human date: ", timeout=240)
        p.sendline("1")
        p.close()


def train_and_infer_helper(
    py_args,
    last_measured_train_time=None,
    last_measured_infer_time=None,
    last_measured_loss=None,
    time_tolerances=0.6,
    infer=True,
):
    """Helper function for running training followed by
       inference tests

    Runs the NMT Tensorflow model with arguments py_arg.  The time taken
    and output are processed and verified against their respective previous
    values.

    Args:
        py_args: dictionary with string keys and values.  Each item
            represents an argument to be used when the model is run
        last_measured_train_time: float representing the previously
            measured time to train the model
        last_measured_infer_time: float representing the previously
            measured time to perform inference on the model
        last_measured_loss: float representing the previously measured
            final loss of the model in training
        time_tolerances:  float representing the percentage tolerance
            on the previously measured values to be asserted against
        infer: bool perform inference if True
    Returns:
        None

    Raises:
        AssertionError: If the measured values are out of the range of the
            previously measured values with the tolerance applied
    """

    out = run_test_helper(
        run_tensorflow_nmt, total_run_time=last_measured_train_time, **py_args
    )

    average_loss = get_results(out)

    if last_measured_loss:
        loss_minimum = get_minimum_with_tolerance(last_measured_loss, 0.2)
        loss_maximum = get_maximum_with_tolerance(last_measured_loss, 0.2)
        assert average_loss >= loss_minimum
        assert average_loss <= loss_maximum

    if infer:
        py_args["--infer"] = ""
        run_test_helper(
            run_tensorflow_nmt, total_run_time=last_measured_infer_time, **py_args
        )


def get_results(output):
    """Finds the average loss, items/sec and tokens/sec in the output and
       returns the final average loss and the average items/sec and
       tokens/sec respectively"""

    line_regex = re.compile(
        r"Step:\s*\d+. Average Loss ([\d.]+). "
    )

    results = parse_results_with_regex(output, line_regex)

    avg_loss_list = results[0]

    return avg_loss_list[-1]
