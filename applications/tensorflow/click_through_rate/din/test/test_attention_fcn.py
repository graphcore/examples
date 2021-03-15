# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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
Tests covering attention used by the DIN model.
"""
import tensorflow as tf
import unittest
import pytest
import numpy as np
import sys
from pathlib import Path

# Add common module to path
common_path = Path(Path(__file__).absolute().parent.parent.parent)
sys.path.append(str(common_path))
from common.utils import din_attention
from din.din_model import DIN

seed = 3
tf.set_random_seed(seed)


@pytest.mark.category1
@pytest.mark.ipus(1)
class TestDINFCN(unittest.TestCase):
    """Testing att layer"""

    @classmethod
    def setUpClass(cls):
        cls.model_dtype = tf.float32
        cls.ATTENTION_SIZE = 1


    def test_att_results(self):
        # test attention layer output

        query_value = np.ones([4, 2], np.float32)
        query_value = query_value * 0.8
        query_inp = tf.placeholder(shape=[4, 2], dtype='float32')

        facts_value = np.ones([4, 8, 2], np.float32)
        facts_value = facts_value * 0.5
        facts_inp = tf.placeholder(shape=[4, 8, 2], dtype='float32')

        mask_value = np.ones([4, 8], np.float32)
        mask_value = mask_value * 0.2
        mask_inp = tf.placeholder(shape=[4, 8], dtype='float32')

        out = din_attention(query_inp, facts_inp, self.ATTENTION_SIZE, mask_inp)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(out, feed_dict={query_inp: query_value, facts_inp: facts_value, mask_inp: mask_value})
        y0 = np.float32(0.5)
        y1 = np.float32(0.5)
        self.assertAlmostEqual(output[0, 0, 0], y0, delta = 0.01)
        self.assertAlmostEqual(output[0, 0, 0], y1, delta = 0.01)


    def test_fcn_results(self):
        # test fcn results

        inputs_value = np.ones([2, 6, 2], np.float32)
        inp = tf.placeholder(shape=[2, 6, 2], dtype='float32')
        y_hat = DIN.build_fcn_net(self, inp)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(y_hat, feed_dict={inp: inputs_value})
        y0 = np.float32(0.5225718)
        y1 = np.float32(0.47742826)
        self.assertAlmostEqual(y[0, 0, 0], y0, delta = 0.01)
        self.assertAlmostEqual(y[0, 0, 1], y1, delta = 0.01)
