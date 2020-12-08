# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from imdb import get_dataset


def pytest_sessionstart(session):
    print("Getting IMDB dataset...")
    get_dataset()
