# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
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


"""
Tests covering various BERT training options.
"""
import glob
import os
import unittest
import pytest

from .run_popdist_train import run_popdist_train


def get_configs():
    """Dynamically read all configs in the test config directory."""
    THIS_MODULE_PATH = os.path.dirname(__file__)
    filenames = glob.glob(os.path.join(THIS_MODULE_PATH, 'configs', '*.json'))
    filenames = [f for f in filenames if "pretrain_tiny" in f]

    return filenames


filenames = get_configs()


def pytest_generate_tests(metafunc):
    if "config" in metafunc.fixturenames:
        metafunc.parametrize("config", filenames, ids=filenames)


@pytest.mark.ipus(2)
class TestBuild(object):
    """Test the build for each config in the directory."""

    def test_build(self, config):
        out = run_popdist_train(**{
                                    '--config': config,
                                    '--replicas': 2,
                                    '--num-train-steps': 10,
                                    '--generated-data': ''
                                    }
                                )
        output = str(out.stdout, 'utf-8')
        print(f"'\nOutput was:\n{output}")
        assert out.returncode == 0
