# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import os
from functools import partial as bind

import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
tf.disable_eager_execution()
tf.disable_v2_behavior()


def custom_product(a, b):
    outputs = {
        "output_types": [tf.float32],
        "output_shapes": [tf.TensorShape([4, 1])],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "libproduct.so")

    return ipu.custom_ops.precompiled_user_op([a, b],
                                              lib_path,
                                              outs=outputs)


def model(use_custom_op, inputs, targets):
    with tf.variable_scope(f"model_vars_{use_custom_op}", reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights", shape=[3, 1],
                                  initializer=tf.zeros_initializer())
        # Forward function:
        preds = custom_product(inputs, weights) if use_custom_op else tf.matmul(inputs, weights)

        # Loss function calculation is not numerically stable but works for this example:
        sigmoid = 0.5 * (tf.math.tanh(preds) + 1)
        probs = sigmoid * targets + (1 - sigmoid) * (1 - targets)
        training_loss = tf.math.reduce_sum(-tf.math.log(probs))

        # Optimiser:
        opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)
        train_op = opt.minimize(training_loss)

        # We can implicitly check correctness of custom op's grad wrt to weights by
        # checking the optimisation result later but if we want to check the gradient
        # w.r.t. the input we need to explicitly ask tensorflow for it (as it has no
        # effect on the optimisation and therefore it is not actually computed):
        gradOfLossWrtInput = tf.gradients(training_loss, [inputs])[0]
        print(f"dL/dI: {gradOfLossWrtInput}")

        return training_loss, weights, gradOfLossWrtInput, train_op


# Values:
input_values = np.array([[0.52, 1.12,  0.77],
                         [0.88, -1.08, 0.15],
                         [0.52, 0.06, -1.30],
                         [0.74, -2.49, 1.39]])
target_values = np.array([[1, 1, 0, 1]]).transpose()

# Variables
with tf.device("cpu"):
    inputs = tf.placeholder(tf.float32, [4, 3])
    targets = tf.placeholder(tf.float32, [4, 1])

with ipu.scopes.ipu_scope("/device:IPU:0"):
    regression = bind(model, False)
    regression_custom = bind(model, True)
    fetches = ipu.ipu_compiler.compile(regression, [inputs, targets])
    fetches_custom = ipu.ipu_compiler.compile(regression_custom, [inputs, targets])

cfg = ipu.utils.create_ipu_config()
cfg = ipu.utils.auto_select_ipus(cfg, 1)
ipu.utils.configure_ipu_system(cfg)
ipu.utils.move_variable_initialization_to_cpu()

# Run the optimisation with the built in op and the
# custom op and record the relevant results:
losses = []
custom_losses = []
grads = []
custom_grads = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        loss, current_weights, dLdI = sess.run(
            fetches, feed_dict={inputs: input_values, targets: target_values})
        losses.append(loss)
        grads.append(dLdI)

    for i in range(100):
        loss, current_weights_custom, dLdI = sess.run(
            fetches_custom, feed_dict={inputs: input_values, targets: target_values})
        custom_losses.append(loss)
        custom_grads.append(dLdI)

# Test that the losses and input gradients matched at
# every step and that the final weights match:
if not np.allclose(custom_losses, losses, equal_nan=True):
    raise RuntimeError("The losses do not match.")

if not np.allclose(grads, custom_grads, equal_nan=True):
    raise RuntimeError("Gradients do not match.")

if not np.allclose(current_weights, current_weights_custom, equal_nan=True):
    raise RuntimeError("The final weights do not match.")

print("Losses, grads and weights match.")
