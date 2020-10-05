# Copyright 2020 Graphcore Ltd.

import os
import subprocess

import examples_tests.test_util as test_util


FILES_TO_GENERATE = [
    "training.csv",
    "validation.csv",
    "human_vocab.json",
    "machine_vocab.json"
]

DATA_GEN_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__))
)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data"
)


def generate_nmt_data():
    """Runs the data generation script which creates the data
       required for the NMT tests."""

    subprocess.run(
        ["./generate_data.sh"],
        universal_newlines=True,
        cwd=DATA_GEN_SCRIPT_PATH
    )

    if not test_util.check_data_exists(
        DATA_PATH, FILES_TO_GENERATE
    ):
        raise Exception(
            f"Dataset generation failed. Cannot find files"
            f" {FILES_TO_GENERATE} in location {DATA_PATH}."
        )

    print("Successfully generated datasets")


def pytest_sessionstart(session):
    """Generate the data required for the tests."""
    generate_nmt_data()


def pytest_sessionfinish(session, exitstatus):
    """Remove data generated for the tests."""
    print("\nRemoving datasets generated for the test session.")
    for file in FILES_TO_GENERATE:
        file_path = os.path.join(DATA_PATH, file)
        if os.path.exists(file_path):
            os.remove(file_path)
