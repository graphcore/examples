# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from pathlib import Path
from subprocess import PIPE, run

import pytest
from examples_tests.execute_once_per_fs import ExecuteOncePerFS

examples_dir = Path(__file__).parent.parent.parent.parent


@pytest.fixture
def ipu_static_ops(scope="session"):
    """This function builds the ipu_static_ops
    library for any tests that rely on it.
    """
    build_path = Path(examples_dir, "gnn", "message_passing", "tensorflow2", "static_ops")

    shared_libs = ["custom_grouped_gather_scatter.so"]
    paths = [Path(build_path, f) for f in shared_libs]

    # Use exclusive lockfile to avoid race conditions on the build:
    lock_path = Path(build_path, ".ipu_static_ops.pytest.build.lockfile")

    @ExecuteOncePerFS(lockfile=lock_path, file_list=paths, timeout=120, retries=20)
    def build_ipustaticops():
        run(["make", "clean"], cwd=build_path)
        run(["make", "-j"], cwd=build_path)

    build_ipustaticops()
