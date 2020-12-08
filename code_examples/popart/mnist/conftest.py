# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
from common import download_mnist


def pytest_sessionstart(session):
    """Load the mnist data at the start of the session"""
    download_mnist(os.path.dirname(__file__))
