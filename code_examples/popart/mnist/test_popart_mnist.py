# Copyright 2019 Graphcore Ltd.
import inspect
import os
import subprocess
import sys
import unittest

import tests.test_util as tu


def run_popart_mnist_training(**kwargs):
    """Helper function to run popart mnist linear model python script with
       command line arguments"""
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    out = tu.run_python_script_helper(cwd, "popart_mnist.py", **kwargs)
    return out


class TestPopARTMNISTImageClassification(unittest.TestCase):
    """High-level integration tests for training with the MNIST data-set"""

    @classmethod
    def setUpClass(cls):
        cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        download_mnist(cwd)
        cls.accuracy_tolerances = 3.0
        cls.generic_arguments = {
            "--batch-size": 4,
            "--batches-per-step": 1000,
            "--epochs": 10,
            "--num-ipus": 1
        }

    def test_mnist_train(self):
        """Generic test on default arguments in training"""
        py_args = self.generic_arguments.copy()
        out = tu.run_test_helper(
            run_popart_mnist_training,
            **py_args
        )
        expected_accuracy = [
            88.88, 89.63, 89.83, 90.01, 90.12, 90.22, 90.40, 90.59, 90.65, 90.70
        ]
        tu.parse_results_for_accuracy(
            out, expected_accuracy, self.accuracy_tolerances
        )

    def test_mnist_train_sharded(self):
        """Generic test on default arguments in training over 2 IPUs"""
        py_args = self.generic_arguments.copy()
        py_args["--num-ipus"] = 2
        out = tu.run_test_helper(
            run_popart_mnist_training,
            **py_args
        )
        expected_accuracy = [
            88.88, 89.63, 89.83, 90.01, 90.12, 90.22, 90.40, 90.59, 90.65, 90.70
        ]
        tu.parse_results_for_accuracy(
            out, expected_accuracy, self.accuracy_tolerances
        )

    def test_mnist_train_sharded_pipelined(self):
        """Generic test on default arguments in training over 2 IPUs
           and pipelined"""
        py_args = self.generic_arguments.copy()
        py_args["--num-ipus"] = 2
        py_args["--pipeline"] = ""
        out = tu.run_test_helper(
            run_popart_mnist_training,
            **py_args
        )
        expected_accuracy = [
            88.11, 88.69, 88.91, 88.94, 88.92, 88.98, 89.05, 89.14, 89.18, 89.25
        ]
        tu.parse_results_for_accuracy(
            out, expected_accuracy, self.accuracy_tolerances
        )

    def test_mnist_all_data(self):
        """Generic test using all the available data (10,000)"""
        py_args = self.generic_arguments.copy()
        py_args["--epochs"] = 2
        py_args["--batch-size"] = 10
        py_args["--batches-per-step"] = 1000
        tu.run_test_helper(
            run_popart_mnist_training,
            **py_args
        )

    def test_mnist_simulation(self):
        """Simulation test with basic arguments"""
        py_args = self.generic_arguments.copy()
        py_args["--simulation"] = ""
        py_args["--epochs"] = 2
        tu.run_test_helper(
            run_popart_mnist_training,
            **py_args
        )

    def test_mnist_log_graph_trace(self):
        """Basic test with log-graph-trace argument"""
        py_args = self.generic_arguments.copy()
        py_args["--log-graph-trace"] = ""
        py_args["--epochs"] = 1
        tu.run_test_helper(
            run_popart_mnist_training,
            **py_args
        )


def download_mnist(file_path):
    """Download the MNIST dataset hosted on GC public S3 bucket"""

    if check_all_data_present(file_path):
        print("MNIST dataset already downloaded, skipping download")
        return

    os.chdir(file_path)
    out = subprocess.check_output(
        "./get_data.sh", env=os.environ.copy(), universal_newlines=True
    )

    if not check_all_data_present(file_path):
        raise OSError("MNIST dataset download unsuccessful")

    print("Successfully downloaded MNIST dataset")


def check_all_data_present(file_path):
    """Checks the data exists in location file_path"""

    filenames = [
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
    ]

    data_path = os.path.join(file_path, "data")

    return tu.check_data_exists(data_path, filenames)


if __name__ == "__main__":
    unittest.main()
