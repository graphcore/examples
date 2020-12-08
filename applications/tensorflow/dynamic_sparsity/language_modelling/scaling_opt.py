# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
This module exposes an Optimizer wrapper to get regular tf.train.Optimizers to
allow for loss scaling and gradient scaling, which can prove useful to fight under and overflows
when working with float16 precision
"""

import os
import tensorflow.compat.v1 as tf

from logging import getLogger
from typing import Type


logger = getLogger(os.path.basename(__file__))


def LossScalingOptimizer(cls: Type[tf.train.Optimizer]) -> Type[tf.train.Optimizer]:
    if not issubclass(cls, tf.train.Optimizer):
        raise ValueError(f'Class {cls} does not inherit from tf.compat.v1.train.Optimizer')

    class Wrapped(cls):
        def __init__(self, *args, loss_scale=1.0, unscale_grad_pre_acc=True, **kwargs):
            super(Wrapped, self).__init__(*args, **kwargs)
            self._scale = loss_scale
            self._unscale_pre_acc = unscale_grad_pre_acc
            self._inv_scale = 1 / loss_scale

        def compute_gradients(self, loss, var_list=None, **kwargs):
            scaled_loss = loss * self._scale
            grads_and_vars = super(Wrapped, self).compute_gradients(
                scaled_loss, var_list=var_list, **kwargs)

            if self._unscale_pre_acc:
                unscaled_grads_and_vars = [(grad * self._inv_scale, var)
                                           for grad, var in grads_and_vars]
            else:
                unscaled_grads_and_vars = grads_and_vars

            return unscaled_grads_and_vars

        def apply_gradients(self, grads_and_vars, global_step=None, name=None):
            if not self._unscale_pre_acc:
                scaled_gradients = [((g * self._inv_scale), v) for g, v in grads_and_vars]
            else:
                scaled_gradients = grads_and_vars

            with tf.control_dependencies([g for g, v in scaled_gradients]):
                apply_updates = super(Wrapped, self).apply_gradients(
                    scaled_gradients, global_step, name)

            return apply_updates

    return Wrapped


def GradScalingOptimizer(cls: Type[tf.train.Optimizer]) -> Type[tf.train.Optimizer]:
    if not issubclass(cls, tf.train.Optimizer):
        raise ValueError(f'Class {cls} does not inherit from tf.compat.v1.train.Optimizer')

    class Wrapped(cls):
        def __init__(self, *args, grad_scale=1.0, scale_grad_pre_acc=True, **kwargs):
            super(Wrapped, self).__init__(*args, **kwargs)
            self._scale = grad_scale
            self._scale_pre_acc = scale_grad_pre_acc

        def compute_gradients(self, loss, var_list=None, **kwargs):
            grads_and_vars = super(Wrapped, self).compute_gradients(
                loss, var_list=var_list, **kwargs)

            if self._scale_pre_acc:
                scaled_grads_and_vars = [(grad * self._scale, var)
                                         for grad, var in grads_and_vars]
            else:
                scaled_grads_and_vars = grads_and_vars

            return scaled_grads_and_vars

        def apply_gradients(self, grads_and_vars, global_step=None, name=None):
            if not self._scale_pre_acc:
                scaled_gradients = [((g * self._scale), v) for g, v in grads_and_vars]
            else:
                scaled_gradients = grads_and_vars

            with tf.control_dependencies([g for g, v in scaled_gradients]):
                apply_updates = super(Wrapped, self).apply_gradients(
                    scaled_gradients, global_step, name)

            return apply_updates

    return Wrapped
