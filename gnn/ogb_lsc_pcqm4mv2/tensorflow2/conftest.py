# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import functools
import os
import time
from pathlib import Path
from subprocess import run

import pytest

base_dir = Path(__file__).parent


def pytest_addoption(parser):
    parser.addoption(
        "--serial",
        action="store_true",
        help="only run serial marked tests",
    )


def pytest_collection_modifyitems(config, items):
    run_only_serial = config.getoption("--serial")
    for item in items:
        if "serial" in item.keywords and not run_only_serial:
            item.add_marker(
                pytest.mark.skip(
                    reason=("This test requires running serially." " Use option --serial to run only serial tests")
                )
            )
        elif "serial" not in item.keywords and run_only_serial:
            item.add_marker(pytest.mark.skip(reason="Only running serial tests."))


@pytest.fixture
def ipu_static_ops(scope="session"):
    """This function builds the ipu_static_ops
    library for any tests that rely on it.
    """
    build_path = Path(base_dir, "static_ops")

    shared_libs = ["custom_grouped_gather_scatter.so"]
    paths = [Path(build_path, f) for f in shared_libs]

    # Use exclusive lockfile to avoid race conditions on the build:
    lock_path = Path(build_path, ".ipu_static_ops.pytest.build.lockfile")

    @ExecuteOncePerFS(lockfile=lock_path, file_list=paths, timeout=120, retries=20)
    def build_ipustaticops():
        run(["make", "clean"], cwd=build_path)
        run(["make", "-j"], cwd=build_path)

    build_ipustaticops()


class ExecuteOncePerFS:
    """Adds synchronization to the execution of a function so it only executes
    once per file-system."""

    def __init__(self, lockfile, file_list, exe_list=[], timeout=60, retries=10):
        self.lockfile = lockfile
        self.file_list = file_list
        self.exe_list = exe_list
        self.timeout = timeout
        self.retries = retries

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            # Race to become master process
            result = None
            try:
                with open(self.lockfile, "x"):
                    # Master process executes function
                    result = fn(*args, **kwargs)
            except FileExistsError:
                pass

            # Every process waits for files to be created
            attempts = 0
            sleep_time = self.timeout / self.retries
            remaining_files = self.file_list[:]
            remaining_exes = self.exe_list[:]
            while attempts < self.retries:
                remaining_files = [path for path in remaining_files if not os.path.exists(path)]
                remaining_exes = [path for path in remaining_exes if not os.access(path, os.R_OK | os.X_OK)]
                if len(remaining_files) == 0 and len(remaining_exes) == 0:
                    return result

                time.sleep(sleep_time)
                attempts += 1

            # If we are here it means that we timed out...
            raise RuntimeError(
                f"Timed out waiting for {remaining_files} to be made" f" and/or {remaining_exes} to become executable."
            )

        return wrapped


def build_global_dependencies():
    # Build globally used objects objects
    build_path = Path(base_dir).joinpath("data_utils/feature_generation")
    shared_libs = ["path_algorithms.so"]
    paths = [Path(build_path, f) for f in shared_libs]

    # Use exclusive lockfile to avoid race conditions on the build:
    lock_path = Path(build_path, ".path_algorithms.pytest.build.lockfile")

    @ExecuteOncePerFS(lockfile=lock_path, file_list=paths, timeout=120, retries=20)
    def build_path_algorithms():
        run(["make", "clean"], cwd=build_path)
        run(["make", "-j"], cwd=build_path)

    build_path_algorithms()


build_global_dependencies()
