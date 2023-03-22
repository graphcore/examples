# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

from pathlib import Path
import pytest

import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).parents[1]


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_run_demo_ipu():
    testing_util.run_command(
        "python3 completed_demos/completed_demo_ipu.py",
        working_path,
        "Program ran successfully",
    )


@pytest.mark.category1
@pytest.mark.ipus(2)
def test_run_demo_faster():
    testing_util.run_command(
        "python3 completed_demos/completed_demo_faster.py",
        working_path,
        "Program ran successfully",
    )


@pytest.mark.category1
@pytest.mark.ipus(2)
def test_run_demo_replication():
    testing_util.run_command(
        "python3 completed_demos/completed_demo_replicated.py",
        working_path,
        "Program ran successfully",
    )


@pytest.mark.category1
@pytest.mark.ipus(4)
def test_run_demo_pipelining():
    testing_util.run_command(
        "python3 completed_demos/completed_demo_pipelining.py",
        working_path,
        "Program ran successfully",
    )
