# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path
from subprocess import run, PIPE
import pytest
import time
import os

public_examples_dir = Path(__file__).parent.parent.parent.parent.parent


@pytest.fixture
def ipu_sparse_ops(scope="session"):
    """This function builds the ipu_sparse_ops
    library for any tests that rely on it.
    """
    build_path = Path(
        public_examples_dir,
        "applications",
        "tensorflow",
        "dynamic_sparsity"
    )
    completed = run(['python-config', '--extension-suffix'], stdout=PIPE)
    extension = completed.stdout.decode().replace('\n', '')
    shared_libs = [f'host_utils{extension}', 'libsparse_matmul.so']
    paths = [Path(build_path, "ipu_sparse_ops", f) for f in shared_libs]

    # Use exclusive lockfile to avoid race conditions on the build:
    lock_path = Path(build_path, ".ipu_sparse_ops.pytest.build.lockfile")
    try:
        with open(lock_path, "x") as lockfile:
            print("\nCleaning dynamic_sparsity")
            run(['make', 'clean'], cwd=build_path)
            print("\nBuilding dynamic_sparsity")
            run(['make', '-j'], cwd=build_path)
    except FileExistsError as e:
        print("\nipu_sparse_ops is already building.")

    exist = [path.exists() for path in paths]
    timeout = 15
    while not all(exist):
        time.sleep(1)
        exist = [path.exists() for path in paths]
        timeout -= 1
        if timeout == 0:
            raise RuntimeError("Timeout waiting for ipu_sparse_ops to build.")
