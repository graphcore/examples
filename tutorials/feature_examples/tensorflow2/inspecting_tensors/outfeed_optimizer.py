# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

"""
Optimizer wrapper that selectively outfeeds gradients
"""

from enum import Enum
from keras.optimizer_v2 import optimizer_v2
from tensorflow.python.training.optimizer import Optimizer


class OutfeedOptimizerMode(Enum):
    """Types used to control the mode of the OutfeedOptimizer.

    These only make a difference to the observed behaviour when using gradient
    accumulation.

    Contains the following values:

    * `AFTER_COMPUTE` - When used with an OutfeedOptimizer,
        the selected gradients will be enqueued after they are
        computed by the wrapped optimizer. When using gradient accumulation
        (for example, when pipelining) this allows the pre-accumulated gradients
        to be inspected.
    * `BEFORE_APPLY` - When used with an OutfeedOptimizer, the selected gradients
        will be enqueued inside the `apply_gradients`
        function before it calls `apply_gradients` on the wrapped optimizer.
        When using gradient accumulation (for example, when pipelining) these
        will be the accumulated gradients.

    """

    AFTER_COMPUTE = "after_compute"
    BEFORE_APPLY = "before_apply"


class OutfeedOptimizer(Optimizer):
    """Optimizer that outfeeds gradients from a wrapped optimizer using a
    MaybeOutfeedQueue.

    The gradients are placed in a dictionary by the MaybeOutfeedQueue;
    each key is the name of the corresponding trainable variable with the
    suffix '_grad'. The gradients are added to the dictionary in reverse
    order - the first entry in the dictionary corresponds to the gradient
    tensor for the last trainable variable.
    """

    def __init__(
        self,
        wrapped_optimizer,
        outfeed_queue,
        outfeed_optimizer_mode=OutfeedOptimizerMode.BEFORE_APPLY,
        name="OutfeedOptimizer",
        model=None,
    ):
        """Construct an OutfeedOptimizer.

        Args:
            wrapped_optimizer: A TensorFlow (derived) optimizer or a Keras
                optimizer_v2.Optimizer. If the latter, then the model must be
                provided.
            outfeed_queue: A MaybeOutfeedQueue to outfeed selected gradients
                (based on the filters supplied when it was created) according
                to the `outfeed_optimizer_mode`. Users of the outfeed_queue
                should only call `outfeed_queue.dequeue()` if `outfeed_queue.enqueued`
                is True.
            outfeed_optimizer_mode: Used to select when gradients will be enqueued.
                This only changes the observed behaviour when using gradient accumulation.
                Defaults to `OutfeedOptimizerMode.BEFORE_APPLY` which means that
                the OutfeedOptimizer will enqueue the selected gradients inside
                the `apply_gradients` function before it calls `apply_gradients`
                on the wrapped optimizer. When using gradient accumulation
                (for example, when pipelining) these will be the accumulated gradients.
                The alternative mode is `OutfeedOptimizerMode.AFTER_COMPUTE`
                which means that the OutfeedOptimizer will enqueue the selected
                gradients in the `compute_gradients` function after they are
                computed by the wrapped optimizer. When using gradient accumulation
                (for example, when pipelining) this allows the pre-accumulated
                gradients to be inspected.
            name: Optional name, defaults to OutfeedOptimizer.
            model: If the wrapped_optimizer is a Keras optimizer_v2.Optimizer
                then the model must be provided.

        Raises:
            ValueError: If the wrapped optimizer is not a subclass of either the
                Keras optimizer_v2.Optimizer or the TensorFlow native
                Optimizer, or if the optimizer is a Keras
                optimizer_v2.Optimizer and a model was not provided.
        """
        super(OutfeedOptimizer, self).__init__(False, name)
        if isinstance(wrapped_optimizer, optimizer_v2.OptimizerV2):
            self._wrapped_optimizer = wrapped_optimizer
            if model is None:
                raise ValueError("The model must be specified if a Keras" " optimizer_v2.Optimizer is being wrapped")
            else:
                self._model = model
        elif isinstance(wrapped_optimizer, Optimizer):
            self._wrapped_optimizer = wrapped_optimizer
        else:
            raise ValueError(
                "Only subclasses of Keras optimizer_v2.Optimizer and TensorFlow" " Optimizer are supported."
            )

        self._outfeed = outfeed_queue
        self._outfeed_optimizer_mode = outfeed_optimizer_mode

    def compute_gradients(self, loss, var_list=None, **kwargs):
        """Compute gradients of "loss" for the variables in "var_list".

        If the `outfeed_optimizer_mode` passed to the constructor was
        `OutfeedOptimizerMode.AFTER_COMPUTE` then this function will enqueue
        a dictionary containing the gradients onto the outfeed queue passed
        to the constructor. The keys are the corresponding variable names with
        "_grad" appended.
        If a filter was provided to the `MaybeOutfeedQueue` then it will be used
        to select which gradients to outfeed - the variable name must include one
        of the filter elements.
        If no gradients are selected then the enqueue function is not called.
        Users of the outfeed queue should only call `outfeed_queue.dequeue()` if
        `outfeed_queue.enqueued` is True.

        Args:
            loss: A Tensor containing the value to minimize.
            var_list: Optional list or tuple of `tf.Variable` to update to minimize
                `loss`.  Defaults to the list of variables collected in the graph
                under the key `GraphKey.TRAINABLE_VARIABLES`.
            **kwargs: Keyword arguments for compute_gradients().

        Returns:
            A list of (gradient, variable) pairs.
        """
        if isinstance(self._wrapped_optimizer, optimizer_v2.OptimizerV2):
            grads = self._wrapped_optimizer.get_gradients(loss, self._model.trainable_weights)
            grads_and_vars = list(zip(grads, self._model.trainable_weights))
        else:
            grads_and_vars = self._wrapped_optimizer.compute_gradients(loss, var_list=var_list, **kwargs)

        if self._outfeed_optimizer_mode == OutfeedOptimizerMode.AFTER_COMPUTE:
            self._maybe_enqueue(grads_and_vars)
        return grads_and_vars

    def _maybe_enqueue(self, grads_and_vars):
        # reverse the order of the gradients
        for g, v in list(reversed(grads_and_vars)):
            self._outfeed.maybe_outfeed(v.name + "_grad", g)
        self._outfeed.maybe_enqueue()

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """Apply gradients to variables.

        If the `outfeed_optimizer_mode` passed to the constructor was
        `OutfeedOptimizerMode.BEFORE_APPLY` then this function will enqueue
        a dictionary containing the gradients onto the outfeed queue passed
        to the constructor. The keys are the corresponding variable names with
        "_grad" appended.
        If a filter was provided to the `MaybeOutfeedQueue` then it will be used
        to select which gradients to outfeed - the variable name must include one
        of the filter elements.
        If no gradients are selected then the enqueue function is not called.
        Users of the outfeed queue should only call `outfeed_queue.dequeue()` if
        `outfeed_queue.enqueued` is True.

        Args:
            grads_and_vars: List of (gradient, variable) pairs as returned by
                compute_gradients().
            global_step: Optional Variable to increment by one after the
                variables have been updated.
            name: Optional name for the returned operation.  Defaults to the
                name passed to the Optimizer constructor.

        Returns:
            An `Operation` that applies the gradients. If `global_step` was not None,
            and the wrapped optimizer is a TensorFlow (derived) optimizer then
            that operation also increments `global_step`.
        """
        if self._outfeed_optimizer_mode == OutfeedOptimizerMode.BEFORE_APPLY:
            self._maybe_enqueue(grads_and_vars)

        if isinstance(self._wrapped_optimizer, optimizer_v2.OptimizerV2):
            return self._wrapped_optimizer.apply_gradients(grads_and_vars)
        else:
            return self._wrapped_optimizer.apply_gradients(grads_and_vars, global_step, name)

    # Override method from tensorflow.python.training.optimizer.Optimizer
    def get_name(self):
        return self._wrapped_optimizer.get_name()

    # Override method from tensorflow.python.training.optimizer.Optimizer
    def get_slot(self, var, name):
        return self._wrapped_optimizer.get_slot(var, name)

    # Override method from tensorflow.python.training.optimizer.Optimizer
    def get_slot_names(self):
        return self._wrapped_optimizer.get_slot_names()

    # Override method from tensorflow.python.training.optimizer.Optimizer
    def variables(self):
        return self._wrapped_optimizer.variables()
