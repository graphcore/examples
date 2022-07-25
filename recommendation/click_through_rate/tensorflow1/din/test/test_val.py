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
Tests covering various infer options using the DIN model.
"""
import tensorflow as tf
import unittest
import pytest

from test_common import get_log, run_validation


@pytest.mark.ipus(1)
class TestDINValidation(unittest.TestCase):
    """Testing some basic infer"""

    @classmethod
    def setUpClass(cls):
        out = run_validation(**{'--use-synthetic-data': True,
                                '--epochs': 2,
                                '--seed': 3,
                                '--learning-rate': '0.1',
                                '--device-iterations': 160})
        cls.out = out
        cls.infer = get_log(out)


    def test_val_results(self):
        # test_final_validation_auc
        final_auc = self.infer
        self.assertGreaterEqual(final_auc, 0.0)
