# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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
import subprocess
import sys
import unittest


def run_resnet(**kwargs):
    cmd = ["python" + str(sys.version_info[0]), 'resnet.py']
    # Flatten kwargs and convert to strings
    args = [str(item) for sublist in kwargs.items() for item in sublist if item != '']
    cmd.extend(args)
    cwd = os.path.dirname(__file__)
    test_env = os.environ.copy()
    test_env["LANG"] = "C.UTF-8"

    try:
        out = subprocess.check_output(
            cmd, cwd=cwd, env=test_env, stderr=subprocess.PIPE).decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(f"TEST FAILED")
        print(f"stdout={e.stdout.decode('utf-8',errors='ignore')}")
        print(f"stderr={e.stderr.decode('utf-8',errors='ignore')}")
        raise
    print(out)

    return out


class TestTensorFlowResNetSyntheticBenchmarks(unittest.TestCase):
    """High-level integration tests for TensorFlow CNN synthetic benchmarks"""

    @classmethod
    def setUpClass(cls):
        pass

    # Resnet inference
    @pytest.mark.ipus(1)
    def test_resnet_18_inference_batch_size_1(self):
        out = run_resnet(**{'--size': 18, '--batch-size': 1})

    @pytest.mark.ipus(1)
    def test_resnet_18_inference_batch_size_16(self):
        out = run_resnet(**{'--size': 18, '--batch-size': 16})

    @pytest.mark.ipus(1)
    def test_resnet_50_inference_batch_size_1(self):
        out = run_resnet(**{'--size': 50, '--batch-size': 1})

    @pytest.mark.ipus(1)
    def test_resnet_50_inference_batch_size_8(self):
        out = run_resnet(**{'--size': 50, '--batch-size': 8})
