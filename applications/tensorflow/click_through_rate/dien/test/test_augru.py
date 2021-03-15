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
Tests covering augru used by the DIEN model.
"""
import tensorflow as tf
import unittest
import pytest
import numpy as np
from tensorflow.python.ops import init_ops
from tensorflow.python.ipu import utils
from tensorflow.python.framework import ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu.ops.rnn_ops import PopnnAUGRU


@pytest.mark.category1
@pytest.mark.ipus(1)
class TestDIENAUGRU(unittest.TestCase):
    """Testing augru layer"""

    @classmethod
    def setUpClass(cls):
        cls.model_dtype = tf.float32
        cls.HIDDEN_SIZE = 2

    """
    Test the augru layer
    The gru layer is implemented by PopnnAUGRU, test it working correctly with
      - inputs, a tensor with dimension: (batch_size, sequence_length, hidden_size)
      - seq_len, a tensor with dimension: (batch_size, )
      - alphas, a tensor with dimension: (sequence_length, batch_size)

    To check the output value as expected
    """
    def test_augru(self):
        seqlen = 3
        bs = 3
        inputs_value = np.ones([bs, seqlen, self.HIDDEN_SIZE], np.float32)
        seq_len_value = np.array([1, 3, 2], np.int32)

        alphas_value = np.ones([seqlen, bs], np.float32)
        alphas_value = alphas_value * 0.5
        inputs = tf.placeholder(shape=[bs, seqlen, self.HIDDEN_SIZE], dtype=self.model_dtype)
        seq_len = tf.placeholder(shape=[bs], dtype=tf.int32)
        alphas = tf.placeholder(shape=[seqlen, bs], dtype=self.model_dtype)

        cfg = utils.create_ipu_config(profiling=False, profile_execution=False)
        cfg = utils.auto_select_ipus(cfg, 1)
        utils.configure_ipu_system(cfg)
        utils.move_variable_initialization_to_cpu()

        with ops.device("/device:IPU:0"):
            train_ipu = ipu_compiler.compile(self.augru_model, inputs=[inputs, seq_len, alphas])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for var in tf.global_variables():
                if var.name == 'popnn_augru/kernel:0':
                    augru_kernel = np.array([[0.3188401, 0.8256132, -0.12287354, 0.8648142, -0.17983055, -0.45415568],
                                            [-0.29249465, 0.65579015, -0.75681853, 0.4331085, -0.07700777, -0.47652483],
                                            [-0.20116574, 0.52735907, -0.08258069, -0.21897888, -0.54514384, 0.32709408],
                                            [-0.43361932, -0.62175727, 0.28278595, 0.13071388, -0.29585528, -0.14058399]])
                    augru_kernel_var = var
            sess.run(tf.assign(augru_kernel_var, augru_kernel))
            outputs_expected = np.array([[[-0.15881832, -0.39365855], [0., 0.], [0., 0.]],
                                        [[-0.15881832, -0.39365855], [-0.1270374, -0.56743807], [-0.09283338, -0.6407641]],
                                        [[-0.15881832, -0.39365855], [-0.1270374, -0.56743807], [0., 0.]]])
            outputs = sess.run(train_ipu, feed_dict={inputs: inputs_value, seq_len: seq_len_value, alphas: alphas_value})
            augru_kernel_updated = sess.run(augru_kernel_var)
            augru_kernel_expected = np.array([[0.31478855, 0.81888944, -0.12453551, 0.863326, -0.40852502, -0.5518727],
                                             [-0.2965462, 0.6490664, -0.7584805, 0.4316203, -0.30570224, -0.5742418],
                                             [-0.20129025, 0.52758944, -0.08233033, -0.21876118, -0.5368969, 0.3306306],
                                             [-0.43399453, -0.6211322, 0.28351453, 0.13140172, -0.25127774, -0.12138209]])
            self.assertAlmostEqual(np.mean(outputs-outputs_expected), np.float32(0.0), delta = 1e-7)
            self.assertAlmostEqual(np.mean(augru_kernel_expected-augru_kernel_updated), np.float32(0.0), delta = 1e-8)


    def augru_model(self, inputs, seq_len, alphas):
        alphas = tf.reshape(alphas, [tf.shape(inputs)[0], -1])
        augru = PopnnAUGRU(self.HIDDEN_SIZE)
        rnn_outputs, _ = augru(inputs, seq_len, alphas, time_major=False)
        rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
        loss = tf.reduce_mean(rnn_outputs - 1.0)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars)
        with tf.control_dependencies([train_op]):
            return rnn_outputs
