# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes


class SGD(tf.train.GradientDescentOptimizer):
    """
    Refactoring of the SGD optimizer with distinct functions for the weight update
    logic and variable assignment.
    """
    def _apply_weight_update(self, grad, var):
        grad = math_ops.cast(grad, var.dtype.base_dtype)
        lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        return var - (lr * grad)

    def _resource_apply_dense(self, grad, var):
        return var.assign(self._apply_weight_update(grad, var))


class Momentum(tf.train.MomentumOptimizer):
    """
    Refactoring of the Momentum optimizer with distinct functions for the weight update
    logic and variable assignment.
    """
    def _apply_weight_update(self, grad, var):
        mom = self.get_slot(var, "momentum")
        grad = math_ops.cast(grad, var.dtype.base_dtype)
        lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        m = math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
        mom = mom.assign(m * mom + grad)
        return var - (lr * mom)

    def _resource_apply_dense(self, grad, var):
        return var.assign(self._apply_weight_update(grad, var))


class RMSProp(tf.train.RMSPropOptimizer):
    """
    Refactoring of the RMSProp optimizer with distinct functions for the weight update
    logic and variable assignment.
    """
    # Ported from source C++ code
    # ms.device(d) += (grad.square() - ms) * (static_cast<T>(1) - rho());
    # mom.device(d) =
    #     mom * momentum() + (grad * lr()) / ((ms + epsilon()).sqrt());
    # var.device(d) -= mom;
    def _apply_weight_update(self, grad, var):
        rms = self.get_slot(var, "rms")
        mom = self.get_slot(var, "momentum")
        grad = math_ops.cast(grad, var.dtype.base_dtype)
        lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        rho = math_ops.cast(self._decay_tensor, var.dtype.base_dtype)
        m = math_ops.cast(self._momentum_tensor, var.dtype.base_dtype)
        eps = math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype)
        if self._centered:
            raise NotImplementedError("Centered RMSProp not implemented.")
        else:
            rms = rms.assign_add((grad ** 2 - rms) * (1 - rho))
            mom = mom.assign(mom * m + (grad * lr) / math_ops.sqrt(rms + eps))
            return var - mom

    def _resource_apply_dense(self, grad, var):
        return var.assign(self._apply_weight_update(grad, var))


def make_fp32_optimiser(optimiser_class):
    """
    Modify optimiser such that an fp32 copy of fp16 weights are created. This means that
    the fp32 version can be offloaded using the variable offload feature. Compared to using fp32
    weights natively this can reduce the live weight memory by a factor of two at the cost of
    increasing the communication cost of retrieving optimiser state from external memory.

    For the default tensorflow optimizers the weight update logic and the update of the
    variable are defined in a single function. This is incompatible with this wrapper as
    it would require a write op followed by a read op of a variable. While these ops could
    be defined, the correct ordering cannot be guaranteed. For this reason optimizers
    with distinct update and write functions are required.

    :param optimiser_class:
    :return:
    """
    class Fp32Optimiser(optimiser_class):
        def __init__(self, *args, **kwargs, ):
            super().__init__(*args, **kwargs)

        def _create_slots(self, var_list):
            new_var_list = []
            for v in var_list:
                if v.dtype.base_dtype != dtypes.float32:
                    new_var_list.append(self._get_or_make_slot(v,
                                                               math_ops.cast(v.initialized_value(),
                                                                             dtypes.float32),
                                                               "fp32", "fp32"))
                else:
                    new_var_list.append(v)
            return super()._create_slots(new_var_list)

        def _apply_weight_update(self, grad, var):
            if var.dtype.base_dtype == dtypes.float32:
                return super()._apply_weight_update(grad, var)
            else:
                orig_var = var
                var = self.get_slot(var, "fp32")
                updated_var = super()._apply_weight_update(
                    math_ops.cast(grad, dtypes.float32), var)
                apply_fp32 = var.assign(updated_var)
                with tf.control_dependencies([apply_fp32]):
                    return math_ops.cast(updated_var, orig_var.dtype.base_dtype)

    return Fp32Optimiser
