# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Optimisers for low precision training on IPU."""

from typing import List, Optional, Tuple

import tensorflow.compat.v1 as tf


class Adam(tf.train.Optimizer):
    """Adam with float16 safety.

    For variables in float32, all computation and storage is in float32.

    For variables in float16, all computation and the variance state `adam_v`
    is in float32, but the momentum state `adam_m` is stored in `float16`.

    Note that this optimizer does not increment global_step as indicated
    by the `tf.train.Optimizer` documentation. This is for easier IPU interop.

    See: "Adam: A Method for Stochastic Optimization", https://arxiv.org/abs/1412.6980.
    """
    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-08,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _get_update(self, variable: tf.Variable, gradient: tf.Tensor,
                    step_size: tf.Tensor) -> List[tf.Operation]:
        with tf.variable_scope(variable.op.name):
            gradient = tf.cast(gradient, tf.float32)
            state_m = tf.get_variable(
                "adam_m",
                shape=variable.shape,
                dtype=variable.dtype,
                initializer=tf.zeros_initializer(),
                trainable=False,
            )
            updated_m = (self.beta1 * tf.cast(state_m, tf.float32) +
                         (1 - self.beta1) * gradient)
            state_v = tf.get_variable(
                "adam_v",
                shape=variable.shape,
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                trainable=False,
            )
            updated_v = self.beta2 * state_v + (1 - self.beta2) * (gradient**2)
            delta = step_size * updated_m / (tf.sqrt(updated_v) + self.epsilon)
            updated_variable = tf.cast(variable, tf.float32) - delta
            return [
                variable.assign(tf.cast(updated_variable, variable.dtype)),
                state_m.assign(tf.cast(updated_m, state_m.dtype)),
                state_v.assign(updated_v),
            ]

    def apply_gradients(
        self,
        grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]],
        global_step: Optional[tf.Tensor],
        name: Optional[str],
    ) -> tf.Operation:
        assert global_step is not None, "Adam requires global_step for bias correction"
        with tf.variable_scope("adam"):
            step = tf.cast(global_step + 1, tf.float32)
            bias_correction = tf.sqrt(1 - self.beta2**step) / (
                1 - self.beta1**step)
            step_size = self.learning_rate * bias_correction
            return tf.group([
                update for gradient, variable in grads_and_vars
                for update in self._get_update(variable, gradient, step_size)
            ])

    def minimize_with_global_step(self, loss: tf.Tensor) -> tf.Operation:
        """As minimize(), but creates and increments global_step."""
        global_step = tf.train.get_or_create_global_step()
        update = self.minimize(loss, global_step=global_step)
        with tf.control_dependencies([update]):
            return tf.group([update, global_step.assign(global_step + 1)])
