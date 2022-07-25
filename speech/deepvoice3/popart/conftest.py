# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
from pathlib import Path
from subprocess import run

import nltk


def get_nltk_data():
    """Gets the NLTK data using the NLTK python module."""
    nltk.download("cmudict")


def pytest_sessionstart(session):
    get_nltk_data()
