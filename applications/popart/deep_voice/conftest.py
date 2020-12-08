# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
from pathlib import Path
from subprocess import run

import nltk


def rebuild_custom_ops():
    """The objective of this script is to:
    1.) Delete the existing custom ops if it exists
    2.) Perform the make command
    3.) Validate a custom_ops.so now does exist"""
    model_path = Path(__file__).resolve().parent
    custom_ops_path = Path(model_path, "custom_ops.so")
    if custom_ops_path.exists():
        print(f"\nDeleting: {custom_ops_path}")
        os.remove(custom_ops_path)
    print("\nBuilding Custom Ops")
    run(["make"], cwd=custom_ops_path.parent)
    assert custom_ops_path.exists()


def get_nltk_data():
    """Gets the NLTK data using the NLTK python module."""
    nltk.download("cmudict")


def pytest_sessionstart(session):
    get_nltk_data()
    rebuild_custom_ops()
