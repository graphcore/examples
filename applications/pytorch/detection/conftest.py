# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
from pathlib import Path
import subprocess


def pytest_sessionstart(session):
    make_path = Path(__file__).parent.resolve()
    subprocess.run(['make'], shell=True, cwd=make_path)
    build_folder_path = os.path.join(make_path, "utils/custom_ops/build")
    assert os.path.isdir(build_folder_path)
