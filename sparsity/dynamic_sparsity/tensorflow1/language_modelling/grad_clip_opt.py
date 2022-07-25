# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
This module exposes an Optimizer wrapper to get regular tf.train.Optimizers to
allow for gradient clipping
"""

import os
import tensorflow.compat.v1 as tf

from logging import getLogger
from typing import Type

tf.disable_eager_execution()
tf.disable_v2_behavior()

logger = getLogger(os.path.basename(__file__))


def GradientClippingOptimizer(cls: Type[tf.train.Optimizer]) -> Type[tf.train.Optimizer]:
    if not issubclass(cls, tf.train.Optimizer):
        raise ValueError(f'Class {cls} does not inherit from tf.compat.v1.train.Optimizer')

    class Wrapped(cls):
        def __init__(self, *args, norm_clip_threshold=2.0, **kwargs):
            super(Wrapped, self).__init__(*args, **kwargs)
            self._norm_clip_threshold = norm_clip_threshold

        def apply_gradients(self, grads_and_vars, global_step=None, name=None):
            clipped_gradients = [(tf.clip_by_norm(g, self._norm_clip_threshold), v) for g, v in grads_and_vars]
            with tf.control_dependencies([g for g, v in clipped_gradients]):
                apply_updates = super(Wrapped, self).apply_gradients(
                    clipped_gradients, global_step, name)

            return apply_updates

    return Wrapped
