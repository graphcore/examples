# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
This module exposes an Optimizer wrapper to get regular tf.train.Optimizers to
allow for selecting the slots FP precision independently of the variable type.
Currently only supports Adam
"""

import os
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.training.optimizer import _var_key
from tensorflow.python.training import slot_creator
from tensorflow.python.training.adam import AdamOptimizer


from typing import Type
from logging import getLogger

tf.disable_v2_behavior()
tf.disable_eager_execution()

logger = getLogger(os.path.basename(__file__))


def SelectableSlotFPFormatOptimizer(cls: Type[tf.train.Optimizer]) -> Type[tf.train.Optimizer]:
    if not issubclass(cls, AdamOptimizer):
        raise ValueError(f'Class {cls} does not inherit from tf.python.training.adam.AdamOptimizer')

    class Wrapped(cls):
        def __init__(self, slots_dtype, force_fp32_weight_update=True, use_nesterov=False, *args, **kwargs):
            self.slots_dtype = tf.as_dtype(slots_dtype)
            self.use_nesterov = use_nesterov
            self.force_fp32_weight_update = force_fp32_weight_update
            super(Wrapped, self).__init__(*args, **kwargs)

        def _zeros_slot(self, var, slot_name, op_name):
            """Find or create a slot initialized with 0.0.
            This is effectively a copy of the original TF optimizer method
            excepts this one allows to pass a dtype to `create_zeros_slot`.
            Args:
              var: A `Variable` object.
              slot_name: Name for the slot.
              op_name: Name to use when scoping the Variable that
                needs to be created for the slot.
            Returns:
              A `Variable` object.
            """
            named_slots = self._slot_dict(slot_name)
            if _var_key(var) not in named_slots:
                new_slot_variable = slot_creator.create_zeros_slot(var, op_name,
                                                                   dtype=self.slots_dtype)
                self._restore_slot_variable(
                    slot_name=slot_name, variable=var,
                    slot_variable=new_slot_variable)
                named_slots[_var_key(var)] = new_slot_variable

            return tf.cast(named_slots[_var_key(var)], var.dtype)

        def _apply_weight_update(self, grad, var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, use_nesterov):
            if self.force_fp32_weight_update:
                # Cast to fp32 for extra precision
                weight_update_dtype = tf.float32
            else:
                weight_update_dtype = var.dtype

            # cast all variables to the same desired dtype for the update
            m_c = tf.convert_to_tensor(tf.cast(m, weight_update_dtype))
            v_c = tf.convert_to_tensor(tf.cast(v, weight_update_dtype))
            var_c = tf.cast(var, weight_update_dtype)
            lr_c = tf.cast(lr, weight_update_dtype)
            beta1_power_c = tf.cast(beta1_power, weight_update_dtype)
            beta2_power_c = tf.cast(beta2_power, weight_update_dtype)
            beta1_c = tf.cast(beta1, weight_update_dtype)
            beta2_c = tf.cast(beta2, weight_update_dtype)
            epsilon_c = tf.cast(epsilon, weight_update_dtype)
            grad_c = tf.cast(grad, weight_update_dtype)

            # correct for the bias of the first and second order moments
            alpha = lr_c * math_ops.sqrt(1 - beta2_power_c) / (1 - beta1_power_c)

            # update the first order moment
            m_t = beta1_c * m_c + (1.0 - beta1_c) * grad_c
            # update the second order moment
            v_t = beta2_c * v_c + (1.0 - beta2_c) * grad_c * grad_c
            # store the moments in the right dtype
            assign_m = tf.assign(m, tf.cast(m_t, self.slots_dtype))
            assign_v = tf.assign(v, tf.cast(v_t, self.slots_dtype))

            # update the variable
            with tf.control_dependencies([assign_m, assign_v]):
                if use_nesterov:
                    return tf.cast(var_c - ((grad_c * (1.0 - beta1_c) + beta1_c * m_t) * alpha) / (math_ops.sqrt(v_t) + epsilon_c), var.dtype)
                else:
                    return tf.cast(var_c - (m_t * alpha) / (math_ops.sqrt(v_t) + epsilon_c), var.dtype)

        def _resource_apply_dense(self, grad, var):
            m = self.get_slot(var, "m")
            v = self.get_slot(var, "v")

            beta1_power, beta2_power = self._get_beta_accumulators()

            return var.assign(
                self._apply_weight_update(
                    grad=grad,
                    var=var,
                    m=m,
                    v=v,
                    beta1_power=beta1_power,
                    beta2_power=beta2_power,
                    lr=self._lr_t,
                    beta1=self._beta1_t,
                    beta2=self._beta2_t,
                    epsilon=self._epsilon_t,
                    use_nesterov=self.use_nesterov))

    return Wrapped
