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
Tests covering gru used by the DIEN model.
"""
import tensorflow as tf
import unittest
import pytest
import numpy as np
from tensorflow.python.ops import init_ops
from tensorflow.python.ipu import utils
from tensorflow.python.framework import ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu.ops.rnn_ops import PopnnDynamicGRU
from tensorflow.python.ipu.config import IPUConfig


@pytest.mark.category1
@pytest.mark.ipus(1)
class TestDIENGRU(unittest.TestCase):
    """Testing gru layer"""

    @classmethod
    def setUpClass(cls):
        cls.model_dtype = tf.float32
        cls.HIDDEN_SIZE = 2

    """
    Test the gru layer
    The gru layer is implemented by PopnnDynamicGRU, test it working correctly with
      - inputs, a tensor with dimension: (batch_size, sequence_length, hidden_size)
      - seq_len, a tensor with dimension: (batch_size, )

    To check the output value as expected
    """
    def test_gru(self):
        seqLen = 2
        bs = 3
        inputs_value = np.array([[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]], np.float32)
        seq_len_value = np.array([1, 2, 2], np.int32)
        inputs = tf.placeholder(shape=[bs, seqLen, self.HIDDEN_SIZE], dtype = self.model_dtype)
        seq_len = tf.placeholder(shape=[bs], dtype=tf.int32)
        cfg = IPUConfig()
        cfg.auto_select_ipus = 1
        cfg.configure_ipu_system()
        utils.move_variable_initialization_to_cpu()
        with ops.device("/device:IPU:0"):
            train_ipu = ipu_compiler.compile(self.gru_model, inputs=[inputs, seq_len])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for var in tf.global_variables():
                if var.name == 'popnn_dynamic_gru/kernel:0':
                    gru_kernel = np.array([[0.36324948, 0.34305102, -0.47945526, 0.29105264, -0.55362725, 0.33607864],
                                          [-0.20881158, 0.79369456, 0.3866263, -0.55099547, 0.41944432, 0.39612126],
                                          [0.48400682, 0.16632384, -0.78809285, 0.47519642, 0.4464376, -0.63623476],
                                          [-0.57933414, -0.29082513, -0.7381171, 0.77089626, -0.24111485, 0.9164796]])
                    gru_kernel_var = var
            sess.run(tf.assign(gru_kernel_var, gru_kernel))
            outputs_expected = np.array([[[-0.03196924, 0.06592286], [-0, 0]], [[-0.03196924, 0.06592286], [-0.06241067, 0.12973404]],
                                        [[-0.03196924, 0.06592286], [-0.06241067, 0.12973404]]])
            outputs = sess.run(train_ipu, feed_dict = {inputs: inputs_value, seq_len: seq_len_value})
            gru_kernel_updated = sess.run(gru_kernel_var)
            gru_kernel_expected = np.array([[0.35011762, 0.37606436, -0.4793783, 0.29105875, -0.6845508, 0.3001622],
                                            [-0.22194342, 0.8267079, 0.38670325, -0.55098933, 0.28852075, 0.36020482],
                                            [0.48412853, 0.16602053, -0.7880953, 0.4751962, 0.4473563, -0.6360037],
                                            [-0.57958513, -0.2901997, -0.73811203, 0.7708967, -0.24294817, 0.9160184]])
            self.assertAlmostEqual(np.mean(outputs-outputs_expected), np.float32(0.0), delta = 1e-7)
            self.assertAlmostEqual(np.mean(gru_kernel_expected-gru_kernel_updated), np.float32(0.0), delta = 1e-8)


    def gru_model(self, inputs, seq_len):
        gru = PopnnDynamicGRU(self.HIDDEN_SIZE)
        rnn_outputs, _ = gru(inputs, seq_len, time_major=False)
        rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
        loss = tf.reduce_mean(rnn_outputs - 1.0)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars)
        with tf.control_dependencies([train_op]):
            return rnn_outputs
