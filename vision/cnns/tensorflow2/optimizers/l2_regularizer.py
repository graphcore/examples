# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from typing import Type


def add_l2_regularization(optimizer_class: Type[tf.keras.optimizers.Optimizer],
                          l2_regularization: float) -> Type[tf.keras.optimizers.Optimizer]:

    class L2Regularizer(optimizer_class):

        def __init__(self, *args, **kwargs):
            super(L2Regularizer, self).__init__(*args, **kwargs)
            self.l2_regularization = l2_regularization

        def _resource_apply_dense(self, grad, var, apply_state):
            return super()._resource_apply_dense(grad + var * self.l2_regularization, var, apply_state)

    return L2Regularizer
