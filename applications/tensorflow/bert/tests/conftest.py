# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
from pathlib import Path
import pytest
import ctypes
from examples_tests.test_util import remote_buffers_available


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "requires_remote_buffers" in item.keywords and not remote_buffers_available():
            item.add_marker(pytest.mark.skip(
                reason="Requires remote buffers to be enabled on this system."))
