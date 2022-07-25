# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
This module exposes an Optimizer wrapper to get regular tf.train.Optimizers to
update the global step if none is provided. This is the only way to update the global
step when using pipelining at the moment
"""

import os
import tensorflow.compat.v1 as tf

from logging import getLogger
from typing import Type

tf.disable_eager_execution()
tf.disable_v2_behavior()

logger = getLogger(os.path.basename(__file__))


def GlobalStepUpdateOptimizer(cls: Type[tf.train.Optimizer]) -> Type[tf.train.Optimizer]:
    if not issubclass(cls, tf.train.Optimizer):
        raise ValueError(f'Class {cls} does not inherit from tf.compat.v1.train.Optimizer')

    class Wrapped(cls):
        def __init__(self, *args, **kwargs):
            super(Wrapped, self).__init__(*args, **kwargs)

        def apply_gradients(self, grads_and_vars, global_step=None, name=None):
            if global_step is None:
                global_step = tf.train.get_or_create_global_step()

            with tf.control_dependencies([global_step]):
                apply_updates = super(Wrapped, self).apply_gradients(
                    grads_and_vars, global_step, name)

            return apply_updates

    return Wrapped
