# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from tensorflow.keras.datasets import cifar10


def pytest_sessionstart(session):
    """Load the cifar10 data at the start of the session"""
    cifar10.load_data()
