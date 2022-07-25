# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

"""
Tests covering various infer options using the DIEN model.
"""
import tensorflow as tf
import unittest
import pytest

from test_common_dien import get_log, run_validation


@pytest.mark.ipus(1)
class TestDIENValidation(unittest.TestCase):
    """Testing some basic infer"""

    @classmethod
    def setUpClass(cls):
        out = run_validation(**{'--use-synthetic-data': True,
                                '--epochs': 0.1,
                                '--seed': 3,
                                '--micro-batch-size': 4,
                                '--learning-rate': '0.1',
                                '--device-iterations': 160})
        cls.out = out
        cls.infer = get_log(out)


    def test_val_auc(self):
        # test_final_validation_auc
        final_auc = self.infer
        self.assertGreaterEqual(final_auc, 0.0)
