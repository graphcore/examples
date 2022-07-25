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

    If you wish to record the dense gradient into an outfeed queue
    so it will be available on the host it can be accomplished like this:
    ```
    prune_and_grow_outfeed_queue = IPUOutfeedQueue()

    ...
    fc_sparse_0 = SparseFcLayer(...)

    output = fc_sparse_0(inputs, compute_prune_and_grow_w=some_tf_condition)
    ...

    optimizer = SparseOptimizer(tf.train.MomentumOptimizer)(
        learning_rate, momentum, name='momentum',
        sparse_layers=[fc_sparse_0, ...],
        dense_gradient_condition=some_tf_condition,
        prune_and_grow_outfeed=prune_and_grow_outfeed_queue)

    ...
    prune_and_grow_op = prune_and_grow_outfeed_queue.dequeue()
    ```
    """
    if not issubclass(cls, tf.train.Optimizer):
        raise ValueError(f'Class {cls} does not inherit from tf.compat.v1.train.Optimizer')

    class Wrapped(cls):
        def __init__(self, *args, sparse_layers=None, dense_gradient_condition=None, prune_and_grow_outfeed=None, **kwargs):
            super(Wrapped, self).__init__(*args, **kwargs)
            self._sparse_layers = sparse_layers or []
            self._sparse_layers = [
                layer
                for layer in self._sparse_layers
                if layer.is_sparse()
            ]
            self._dense_gradient_condition = dense_gradient_condition
            self._prune_and_grow_outfeed = prune_and_grow_outfeed
            self._dense_grads_and_vars = None  # Initialised by compute_gradients

            # If python boolean and always False, disable it completely
            if isinstance(self._dense_gradient_condition, bool) and not self._dense_gradient_condition:
                self._dense_gradient_condition = None

            if self._dense_gradient_condition is not None and prune_and_grow_outfeed is None:
                raise ValueError('If dense_gradient_condition is set a prune_and_grow_outfeed queue must be provided.')

        def apply_gradients(self, grads_and_vars, global_step=None, name=None):
            # Unconditionally apply updates:
            apply_updates = super(Wrapped, self).apply_gradients(grads_and_vars, global_step, name)

            # If we don't need the dense grad or prune and grow data then we are done:
            logger.debug(f"Optimizer conditionals: dense grads exist: {self._dense_grads_and_vars is not None} "
                         f"prune and grow condition tensor set: {self._dense_gradient_condition is not None}")
            build_conditional_dense_ops = self._dense_grads_and_vars and self._dense_gradient_condition is not None
            if not build_conditional_dense_ops:
                return apply_updates

            # We need to conditionally compute dense grad and return all the variables
            # needed for prune and grow on the very last iteration:
            outfeed_dict = {}
            dense_grads = {
                # Change var name to grad name
                var.name.replace('dummy_dense_weights:0', 'grad_w'): grad
                for grad, var in self._dense_grads_and_vars
            }
            # Sparse weights are returned with the dense grad so that prune and grow
            # algorithms that run on the host have access to both from the same feed:
            sparse_weight_vars = [layer.get_values_var() for layer in self._sparse_layers]
            sparse_weights = {
                var.name: var for var in sparse_weight_vars
            }

            # Record any slot variables that might have to change along with the non-zero values
            slot_names = self.get_slot_names()
            logger.debug(f"Optimiser {self} slot names: {slot_names}")
            slot_vars = {}
            for slot_name in slot_names:
                for layer in self._sparse_layers:
                    slot_var = layer.record_slot_var(slot_name=slot_name, optimizer=self)
                    slot_vars[slot_var.name] = slot_var

            outfeed_dict = {**dense_grads, **sparse_weights, **slot_vars}
            logger.debug(f"Tensors to be enqueued for prune and grow: {outfeed_dict.keys()}")

            dense_update_op = tf.cond(
                tf.convert_to_tensor(self._dense_gradient_condition),
                true_fn=lambda: self._prune_and_grow_outfeed.enqueue(outfeed_dict),
                false_fn=lambda: tf.no_op())

            return tf.group([apply_updates, dense_update_op])

        def compute_gradients(
                self,
                loss,
                var_list=None,
                gate_gradients=tf.train.Optimizer.GATE_OP,
                aggregation_method=None,
                colocate_gradients_with_ops=False,
                grad_loss=None):
            # Locate the dense grad variables
            dense_dummy_vars = []
            for layer in self._sparse_layers:
                dense_dummy_vars.append(
                    layer.get_dense_dummy_var())

            if var_list is None:
                var_list = (
                    tf.trainable_variables() +
                    tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES) +
                    dense_dummy_vars)

            # Fetch grads and vars from super-class
            grads_and_vars = super(Wrapped, self).compute_gradients(
                loss,
                var_list=var_list,
                gate_gradients=gate_gradients,
                aggregation_method=aggregation_method,
                colocate_gradients_with_ops=colocate_gradients_with_ops,
                grad_loss=grad_loss)

            def is_dense_dummy_var(v):
                return any(
                    v is dense_dummy_var
                    for dense_dummy_var in dense_dummy_vars)

            # Intercept and store dense grads and vars if necessary
            if self._dense_gradient_condition is not None:
                # Get only dense grads and vars
                self._dense_grads_and_vars = [
                    (grad, var)
                    for grad, var in grads_and_vars
                    if is_dense_dummy_var(var)
                ]
            else:
                self._dense_grads_and_vars = []

            # Fetch the regular gradients by filtering out the dense variables
            grads_and_vars = [
                (grad, var)
                for grad, var in grads_and_vars
                if not is_dense_dummy_var(var)
            ]

            return grads_and_vars

    return Wrapped
