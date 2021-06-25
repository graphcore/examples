# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from filelock import FileLock
from contextlib import contextmanager


@contextmanager
def lock(lock_path):
    with FileLock(lock_path):
        yield
