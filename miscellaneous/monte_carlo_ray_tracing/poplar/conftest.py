# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path
from subprocess import run, PIPE
from examples_tests.execute_once_per_fs import ExecuteOncePerFS

import pytest

root_dir = Path(__file__).parent
build_dir = root_dir.joinpath('test_cmake_build')


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
        # Get the light repo if doesn't exist already
        light_dir = root_dir / Path("light")
        if not light_dir.exists():
            run(['git', 'clone', 'https://github.com/mpups/light.git'], cwd=root_dir)
            run(['git', 'checkout', '589b32f'], cwd=light_dir)
        build_dir.mkdir(exist_ok=True)
        run(['cmake', '..', '-GNinja'], cwd=build_dir)
        run(['ninja', '-j8'], cwd=build_dir)

    cmake_build()
