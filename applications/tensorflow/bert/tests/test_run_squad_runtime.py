# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
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
Tests covering SQuAD inference using the embedded runtime framework.
"""
import glob
import os
import unittest
import pytest
import re

from .run_train import run_embedded_runtime_squad


def get_configs():
    """Dynamically read all configs in the test config directory."""
    THIS_MODULE_PATH = os.path.dirname(__file__)
    filenames = glob.glob(os.path.join(THIS_MODULE_PATH, 'configs', '*.json'))
    filenames = [f for f in filenames if "squad_large_inference" in f]
    return filenames


filenames = get_configs()


def pytest_generate_tests(metafunc):
    if "config" in metafunc.fixturenames:
        metafunc.parametrize("config", filenames, ids=filenames)


@pytest.mark.ipus(1)
@pytest.mark.requires_remote_buffers
@pytest.mark.ipu_version("ipu2")
class TestBuild(object):
    """Test the build for each config in the directory."""

    def test_build(self, config):
        path = os.path.dirname(os.path.realpath(__file__))
        out = run_embedded_runtime_squad(**{'--config': config,
                                            '--gradient-accumulation-count': 40,
                                            '--batches-per-step': 20,
                                            '--embedded-runtime': '',
                                            '--batch-size': 2,
                                            '--vocab-file': f'{path}/vocab.txt',
                                            '--generated-data': ''})
        output = str(out.stdout, 'utf-8')
        print(f"'\nOutput was:\n{output}")
        assert out.returncode == 0, "Build failed"  # That the build + run works correctly
