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

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.training import training_ops


def batch_norm(x,
               center=True,
               scale=True,
               training=True,
               trainable=True,
               epsilon=1e-6,
               gamma_initializer = tf.ones_initializer(),
               beta_initializer = tf.zeros_initializer(),
               ):
    """Batch Norm function that is compatible with pipelining.

    The normal batch norm function does not work correctly with pipelining as it relies
    on assign ops in the forward pass to update the moving averages which are not allowed.

    This function instead represents the moving averages as trainable variables but with
    a custom gradient that defines its gradient as the moving average update step. This
    means they can be correctly accumulated over the pipeline micro-batches.

    To ensure the moving average updates are correctly applied the Optimizer class must be
    augmented with the 'add_bn_moving_average_updates' function.

    Args:
      x: A Tensor with at least 2 dimensions in NHWC format. All
       shape dimensions must be fully defined.
      center: If True, add offset of `beta` to normalized tensor. If False, `beta`
        is ignored.
      scale: If True, multiply by `gamma`. If False, `gamma` is
        not used. When the next layer is linear (also e.g. `nn.relu`), this can be
        disabled since the scaling can be done by the next layer.
      epsilon: Small float added to variance to avoid dividing by zero.
      training: Whether this is operation is being used in a training network.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      gamma_initializer: Optional initializer for  gamma.
      beta_initializer: Optional initializer for beta.

    Returns:
      A `Tensor` representing the output of the operation.
    """
    with tf.variable_scope('batch_norm'):
        num_channels = x.get_shape().as_list()[3]

        if center:
            beta = tf.get_variable('beta', shape=(num_channels), dtype=x.dtype,
                                   initializer=beta_initializer, trainable=trainable)
        else:
            beta = tf.constant(0.0, shape=(num_channels), dtype=x.dtype)

        if scale:
            gamma = tf.get_variable('gamma', shape=(num_channels), dtype=x.dtype,
                                    initializer=gamma_initializer, trainable=trainable)
        else:
            gamma = tf.constant(1.0, shape=(num_channels), dtype=x.dtype)

        moving_mean = tf.get_variable('moving_mean', shape=(num_channels), dtype=x.dtype,
                                      initializer=tf.zeros_initializer(), trainable=trainable)

        moving_variance = tf.get_variable('moving_variance', shape=(num_channels), dtype=x.dtype,
                                          initializer=tf.ones_initializer(), trainable=trainable)

        if training:
            x, mean, variance = tf.nn.fused_batch_norm(
                x, gamma, beta, epsilon=epsilon, data_format='NHWC')
        else:
            x, mean, variance = tf.nn.fused_batch_norm(
                    x,
                    gamma,
                    beta,
                    mean=moving_mean,
                    variance=moving_variance,
                    epsilon=epsilon,
                    is_training=False,
                    data_format='NHWC')

        @tf.custom_gradient
        def moving_avg_updates(X, moving_m, moving_v):
            def bw(dx):
                return dx, moving_m - mean, moving_v - variance
            return X, bw

        x = moving_avg_updates(x, moving_mean, moving_variance)

    return x


def add_bn_moving_average_updates(optimiser_class, momentum=None):
    class Optimiser(optimiser_class):
        def __init__(self, *args, **kwargs,):
            super().__init__(*args, **kwargs)
            if momentum:
                self._update_step = 1 - momentum

        def _create_slots(self, var_list):
            return super()._create_slots([v for v in var_list if 'batch_norm/moving_' not in v.name])

        def _resource_apply_dense(self, grad, var):
            if 'batch_norm/moving_' in var.name:
                # pose update as a gradient descent update to ensure compatibility with
                # gradient accumulation v1 op
                return training_ops.resource_apply_gradient_descent(
                    var.handle, math_ops.cast(self._update_step, grad.dtype.base_dtype),
                    grad, use_locking=self._use_locking)
            else:
                return super()._resource_apply_dense(grad, var)

    return Optimiser
