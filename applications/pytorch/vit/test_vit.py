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

import inspect
import os
import re
import subprocess
import unittest
import pytest


def run_vit_cifar10(**kwargs):
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    cmd = ["python", "train.py", "--config", "b16_cifar10", "--training-steps", "300"]
    try:
        out = subprocess.check_output(
            cmd, cwd=cwd, stderr=subprocess.PIPE).decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(f"TEST FAILED")
        print(f"stdout={e.stdout.decode('utf-8',errors='ignore')}")
        print(f"stderr={e.stderr.decode('utf-8',errors='ignore')}")
        raise
    return out


class TestViT(unittest.TestCase):

    @pytest.mark.ipus(1)
    def test_final_training_loss(self):
        out = run_vit_cifar10()
        loss = 100.0

        for line in out.split("\n"):
            if line.find("Step: 199/") != -1:
                loss = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-2])
                break
        self.assertGreater(loss, 0.1)
        self.assertLess(loss, 0.91)
