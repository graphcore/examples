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


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from functools import partial as bind
from optparse import OptionParser

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu

import ipu_optimizer as _opt
import hardcoded

tf.disable_eager_execution()
tf.disable_v2_behavior()

parser = OptionParser()
parser.add_option('--optimiser', action="store", default='sgd',
                  dest='optimiser')
(opts, args) = parser.parse_args()

expected_losses = hardcoded.expected_losses
expected_grads = hardcoded.expected_grads
expetced_weights = hardcoded.expetced_weights


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
    if use_custom_op == 'sgd':
        opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)
    elif use_custom_op == 'momentum':
        opt = tf.train.MomentumOptimizer(learning_rate=0.05, momentum=0.9,
                                         use_nesterov=False)
    elif use_custom_op == 'lamb':
        opt = _opt.LAMBOptimizer(0.05, loss_scaling=1.0, high_precision=False)
    elif use_custom_op == 'adamw':
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
    regression = bind(model, opts.optimiser)
    fetches = ipu.ipu_compiler.compile(regression, [inputs, targets])

cfg = ipu.utils.create_ipu_config()
cfg = ipu.utils.auto_select_ipus(cfg, 1)
ipu.utils.configure_ipu_system(cfg)
ipu.utils.move_variable_initialization_to_cpu()

# Run the optimisation with each optimiser and compare to known results.
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
if loss > expected_losses[opts.optimiser]:
    raise RuntimeError("The losses has not reached the threshold.")
# Check that the final grads are as expected
if not np.allclose(dLdI, expected_grads[opts.optimiser], equal_nan=True,
                   rtol=1e-4, atol=1e-4):
    raise RuntimeError("Gradients do not match.")
# Check the final weights are unchanged
if not np.allclose(current_weights, expetced_weights[opts.optimiser],
                   equal_nan=True, rtol=1e-3, atol=1e-3):
    raise RuntimeError("The final weights do not match. Currenct Weights {}, Expected weights {}".format(current_weights, expetced_weights[opts.optimiser]))
print("Losses, grads and weights match.")
