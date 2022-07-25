# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Graphcore Ltd.
import re
import logging
import tensorflow as tf
from tensorflow.python import ipu


class AdamLossScalingOptimizer(tf.compat.v1.train.Optimizer):
    """A basic Adam optimizer that includes loss scaling."""
    def __init__(self,
                 learning_rate,
                 loss_scaling,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 name="AdamLossScalingOptimizer",
                 weights_dtype=tf.float16):
        """Constructs a AdamLossScalingOptimizer."""
        super(AdamLossScalingOptimizer, self).__init__(False, name)

        self.learning_rate = tf.cast(learning_rate, dtype=weights_dtype)
        self.loss_scaling = loss_scaling
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.compat.v1.get_variable(name=param_name + "/adam_m",
                                          shape=param.shape.as_list(),
                                          dtype=tf.float32,
                                          trainable=False,
                                          initializer=tf.zeros_initializer())

            v = tf.compat.v1.get_variable(name=param_name + "/adam_v",
                                          shape=param.shape.as_list(),
                                          dtype=tf.float32,
                                          trainable=False,
                                          initializer=tf.zeros_initializer())

            @ipu.outlined_function(unique_sharding=True)
            def grad_fn(grad, param, m, v):
                cast_grad = tf.cast(grad, dtype=tf.float32)
                cast_grad = cast_grad / self.loss_scaling

                # Standard Adam update.
                next_m = (tf.multiply(self.beta_1, m) +
                          tf.multiply(1.0 - self.beta_1, cast_grad))
                next_v = (tf.multiply(self.beta_2, v) +
                          tf.multiply(1.0 - self.beta_2, tf.square(cast_grad)))

                update = tf.cast(next_m / (tf.sqrt(next_v) + self.epsilon),
                                 param.dtype)
                update_with_lr = tf.cast(self.learning_rate, update.dtype) * update
                next_param = param - update_with_lr

                return next_param, next_v, next_m

            next_param, next_v, next_m = grad_fn(grad, param, m, v)

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])

        return tf.group(*assignments, name=name)


    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
