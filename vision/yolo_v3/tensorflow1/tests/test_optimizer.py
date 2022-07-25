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


import os
import sys
from functools import partial as bind
from optparse import OptionParser

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ipu_optimizer as _opt
import tensorflow.compat.v1 as tf
from optimizer_fixture import *
from tensorflow.python import ipu


tf.disable_eager_execution()
tf.disable_v2_behavior()


@pytest.mark.parametrize("optim", ["momentum",
                                   "adamw"])
def test_regression_optimizer(optim, expected_losses, expected_grads, expetced_weights):

    def model(use_custom_op, inputs, targets):
        weights = tf.get_variable("weights", shape=[3, 1],
                                  initializer=tf.zeros_initializer(),
                                  dtype=tf.float16)
        # Forward function:
        preds = tf.matmul(inputs, weights)

        sigmoid = 0.5 * (tf.math.tanh(preds) + 1)
        probs = sigmoid * targets + (1 - sigmoid) * (1 - targets)
        training_loss = tf.math.reduce_sum(-tf.math.log(probs))

        gradOfLossWrtInput = tf.gradients(training_loss, [inputs])[0]

        # Optimiser:
        if use_custom_op == "momentum":
            opt = _opt.MomentumOptimizer(learning_rate=0.05, momentum=0.9)
        elif use_custom_op == "adamw":
            opt = _opt.AdamWeightDecayOptimizer(0.05, loss_scaling=1.0)
        train_op = opt.minimize(training_loss)

        return training_loss, weights, gradOfLossWrtInput, train_op

    # Values:
    input_values = np.array([[0.52, 1.12,  0.77],
                             [0.88, -1.08, 0.15],
                             [0.52, 0.06, -1.30],
                             [0.74, -2.49, 1.39]])
    target_values = np.array([[1, 1, 0, 1]]).transpose()

    # Variables
    with tf.device("cpu"):
        inputs = tf.placeholder(tf.float16, [4, 3])
        targets = tf.placeholder(tf.float16, [4, 1])
    with ipu.scopes.ipu_scope("/device:IPU:0"):
        regression = bind(model, optim)
        fetches = ipu.ipu_compiler.compile(regression, [inputs, targets])

    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.configure_ipu_system()

    # Run the optimisation with each optimizer and compare to known results.
    losses = []
    custom_losses = []
    lamb_losses = []
    grads = []
    custom_grads = []
    lamb_grads = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            loss, current_weights, dLdI = sess.run(
                fetches, feed_dict={inputs: input_values, targets: target_values})
            losses.append(loss)
            grads.append(dLdI)

    # Check that the loss is below the threshold given
    if loss > expected_losses[optim]:
        raise RuntimeError("The losses has not reached the threshold.")
    # Check that the final grads are as expected
    if not np.allclose(dLdI, expected_grads[optim], equal_nan=True,
                       rtol=1e-4, atol=1e-4):
        raise RuntimeError("Gradients do not match.")
    # Check the final weights are unchanged
    if not np.allclose(current_weights, expetced_weights[optim],
                       equal_nan=True, rtol=1e-3, atol=1e-3):
        raise RuntimeError("The final weights do not match. Currenct Weights {}, Expected weights {}".format(
            current_weights, expetced_weights[optim]))
    print("Losses, grads and weights match.")
    # clear graph incase there's multiple runs, and we get duplicated ops and variables
    tf.reset_default_graph()
