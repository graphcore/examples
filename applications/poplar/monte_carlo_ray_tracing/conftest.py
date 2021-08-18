# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path
from subprocess import run, PIPE

import functools
import os
import tempfile
import time
import shutil
import subprocess

import pytest

root_dir = Path(__file__).parent
build_dir = root_dir.joinpath('test_cmake_build')


class ExecuteOncePerFS:
    """Adds synchronization to the execution of a function so it only executes
    once per file-system."""

    def __init__(self, lockfile, file_list, exe_list, timeout, retries=10):
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
            sleep_time = self.timeout/self.retries
            remaining_files = self.file_list[:]
            remaining_exes = self.exe_list[:]
            while attempts < self.retries:
                remaining_files = [
                    path for path in remaining_files if not os.path.exists(path)]
                remaining_exes = [
                    path for path in remaining_exes if not os.access(path, os.R_OK | os.X_OK)]
                if len(remaining_files) == 0 and len(remaining_exes) == 0:
                    return result

                time.sleep(sleep_time)
                attempts += 1

            # If we are here it means that we timed out...
            raise RuntimeError(f"Timed out waiting for {remaining_files} to be made and/or {remaining_exes} to become executable.")
        return wrapped


@pytest.fixture
def cmake_build(scope="session"):
    """This function builds the application once for any tests that rely on it."""

    # Use exclusive lockfile to avoid race conditions on the build:
    lock_path = Path(root_dir, ".cmake.pytest.build.lockfile")
    files = ['ipu_trace', 'codelets.gp']
    paths = [Path(build_dir, f) for f in files]
    exes = [Path(build_dir, 'ipu_trace')]

    @ExecuteOncePerFS(lockfile=lock_path, file_list=paths, exe_list=exes, timeout=120, retries=20)
    def cmake_build():
        build_dir.mkdir(exist_ok=True)
        run(['cmake', '..', '-GNinja'], cwd=build_dir)
        run(['ninja', '-j8'], cwd=build_dir)

    cmake_build()
