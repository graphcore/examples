# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
This module exposes an Optimizer wrapper to get regular tf.train.Optimizers to
work with the sparse layer implementations on the IPU.

If the underlying sparse weight matrix has its adjacency pattern manipulated
dynamically then the various optimizer slots made to mirror it have to be
altered as well.

The :decorator:`~ipu_sparse_ops.optimizers.SparseOptimizer` decorator handles
registering any potential optimizer specific slots with the underlying sparse
layer.
"""

import os
import tensorflow.compat.v1 as tf

from logging import getLogger
from typing import Type


logger = getLogger(os.path.basename(__file__))


def SparseOptimizer(cls: Type[tf.train.Optimizer]) -> Type[tf.train.Optimizer]:
    """Decorator for optimizers with sparse support.

    This class decorator wraps the underlying class so that slots are
    registered automatically with potential sparse layers like
    :class:`~sparse_ipu_ops.layers.SparseFcLayer`.

    Example usage:
    ```
    from sparse_ipu_ops.layers import SparseFcLayer
    from sparse_ipu_ops.optimizers import SparseOptimizer

    fc_sparse_0 = SparseFcLayer(...)

    ...

    optimizer = SparseOptimizer(tf.train.MomentumOptimizer)(
        learning_rate, momentum, name='momentum',
        sparse_layers=[fc_sparse_0, ...])
    train_op = optimizer.minimize(loss)
    ```
    """
    if not issubclass(cls, tf.train.Optimizer):
        raise ValueError(f'Class {cls} does not inherit from tf.compat.v1.train.Optimizer')

    class Wrapped(cls):
        def __init__(self, *args, sparse_layers=None, **kwargs):
            super(Wrapped, self).__init__(*args, **kwargs)
            self._sparse_layers = sparse_layers or []
            self._sparse_layers = [
                layer
                for layer in self._sparse_layers
                if layer.is_sparse()
            ]

        def apply_gradients(self, grads_and_vars, global_step=None, name=None):
            apply_updates = super(Wrapped, self).apply_gradients(grads_and_vars, global_step, name)

            for slot_name in self.get_slot_names():
                for layer in self._sparse_layers:
                    logger.debug('Recording slot variable %s for sparse layer %s', slot_name, layer.name)
                    layer.record_slot_var(slot_name=slot_name, optimizer=self)

            return apply_updates

    return Wrapped
