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

import os
import pytest
from tests.utils import bert_root_dir
from examples_utils import load_lib


@pytest.fixture
def custom_ops():
    load_lib(os.path.join(bert_root_dir(), 'custom_ops', 'custom_ops.cpp'))


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "requires_config" in item.keywords and not config.getoption("--config-path"):
            item.add_marker(pytest.mark.skip(
                reason="Requires a config-graph path to run"))
        if "requires_frozen" in item.keywords and not config.getoption("--frozen-path"):
            item.add_marker(pytest.mark.skip(
                reason="Requires a frozen-graph path to run"))


def pytest_addoption(parser):
    parser.addoption(
        "--config-path",
        action="store",
        help="Path to a file containing the BERT configuration parameters.")

    parser.addoption(
        "--frozen-path",
        action="store",
        help="Path to the frozen graph file.")

    parser.addoption(
        "--chkpt-task",
        type="choice",
        choices=("PRETRAINING, SQUAD"),
        default="PRETRAINING",
        help="Checkpoint task - Pretraining or SQuAD.")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_config: Skip if fixture config_path not provided")
    config.addinivalue_line(
        "markers", "requires_frozen: Skip if fixture frozen_path not provided")


@pytest.fixture
def chkpt_path(request):
    return request.config.getoption("--chkpt-path")


@pytest.fixture
def chkpt_task(request):
    return request.config.getoption("--chkpt-task")


@pytest.fixture
def frozen_path(request):
    return request.config.getoption("--frozen-path")
