# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from typing import Type


def add_loss_scaling_to_optimizer(
    optimizer_class: Type[tf.keras.optimizers.Optimizer], loss_scale: float
) -> Type[tf.keras.optimizers.Optimizer]:
    class LossScaleOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(LossScaleOptimizer, self).__init__(*args, **kwargs)

        def _scale_loss(self, loss):
            return loss * loss_scale

        def get_gradients(self, loss, vars):
            loss = self._scale_loss(loss)
            return super().get_gradients(loss, vars)

        def _resource_apply_dense(self, grad, var, apply_state):
            return super()._resource_apply_dense(grad / loss_scale, var, apply_state)

    return LossScaleOptimizer
