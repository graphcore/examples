# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import ctypes
import subprocess
from pathlib import Path
import pytest
from util.log import logger


def pytest_sessionstart(session):
    try:
        subprocess.check_output("sh make_remap.sh", shell=True, cwd=Path(__file__).parent.parent.resolve())
    except subprocess.CalledProcessError as e:
        logger.info(f"Make custom op FAILED")
        logger.info(f"stdout={e.stdout.decode('utf-8',errors='ignore')}")
        logger.info(f"stderr={e.stderr.decode('utf-8',errors='ignore')}")
        raise
