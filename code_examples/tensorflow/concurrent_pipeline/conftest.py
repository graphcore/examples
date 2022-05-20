# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path
from subprocess import run, PIPE
from examples_tests.execute_once_per_fs import ExecuteOncePerFS

import functools
import os
import tempfile
import time
import shutil
import subprocess

import pytest

public_examples_dir = Path(__file__).parent.parent.parent.parent
build_dir = Path(__file__).parent


@pytest.fixture
def custom_ops(scope="session"):
    """This function builds the ipu_sparse_ops
    library for any tests that rely on it.
    """
    build_path = Path(
        public_examples_dir,
        "code_examples",
        "tensorflow",
        "concurrent_pipeline"
    )

    shared_libs = ['libconcurrent_ops.so']
    paths = [Path(build_path, "custom_ops", f) for f in shared_libs]

    # Use exclusive lockfile to avoid race conditions on the build:
    lock_path = Path(build_path, ".custom_ops.pytest.build.lockfile")

    print(f"Building paths: {paths}")

    @ExecuteOncePerFS(lockfile=lock_path, file_list=paths, timeout=120, retries=20)
    def build_ops():
        run(['make', 'clean'], cwd=build_path)
        run(['make', '-j'], cwd=build_path)

    build_ops()
