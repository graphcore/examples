# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from pathlib import Path
from subprocess import run, PIPE
from examples_tests.execute_once_per_fs import ExecuteOncePerFS

import subprocess
import pytest

examples_dir = Path(__file__).parent.parent.parent.parent.parent


@pytest.fixture
def length_regulator_op(scope="session"):
    """This function builds the length_regulator_op
    library for any tests that rely on it.
    """
    build_path = Path(examples_dir, "speech", "fastspeech2",
                      "tensorflow2", "custom_op", "length_regulator")
    shared_libs = ['liblengthRegulator.so']
    paths = [Path(build_path, f) for f in shared_libs]

    # Use exclusive lockfile to avoid race conditions on the build:
    lock_path = Path(build_path, ".length_regulator_op.pytest.build.lockfile")

    @ExecuteOncePerFS(lockfile=lock_path,
                      file_list=paths,
                      timeout=120,
                      retries=20)
    def build_length_regulator():
        run(['make', 'clean'], cwd=build_path)
        run(['make', '-j'], cwd=build_path)

    build_length_regulator()
