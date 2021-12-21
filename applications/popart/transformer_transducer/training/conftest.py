# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
import shutil
from subprocess import run


def rebuild_custom_ops():
    """The objective of this function is to:
    1.) Delete the existing build dir for custom ops if it exists
    2.) Perform the make command
    3.) Validate the build_dir does exist after make"""
    wdir = os.path.join(os.path.dirname(__file__), "..")
    build_dir = os.path.join(wdir, "build")
    if os.path.exists(build_dir):
        print(f"\nRemoving build dir: {build_dir}")
        shutil.rmtree(build_dir)
    print("\nBuilding all Custom Ops")
    run(["make"], cwd=wdir)
    assert os.path.exists(build_dir)


def pytest_sessionstart(session):
    rebuild_custom_ops()
