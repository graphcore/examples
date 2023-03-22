# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from tensorflow_addons.optimizers.weight_decay_optimizers import extend_with_decoupled_weight_decay

from ipu_tensorflow_addons.keras.optimizers import AdamIpuOptimizer, LAMBIpuOptimizer


ALLOWED_OPTIMIZERS = ["AdamW", "LAMB"]


class StaticLossScaleOptimizer(LossScaleOptimizer):
    """
    Modification of the Keras LossScaleOptimizer that only
    supports a static loss scaling. It ensures the gradients
    are in float32 before unscaling and applies this in the
    inner optimizer's gradient_transformers, which we have
    observed tends to be more memory efficient.
    """

    def __init__(self, inner_optimizer, loss_scaling):
        """
        Constructs loss scale optimizer.
        :param inner_optimizer: The optimizer to wrap with this
        functionality.
        :param loss_scaling: Constant loss scaling factor to be applied.
        """
        super().__init__(inner_optimizer, dynamic=False, initial_scale=loss_scaling, dynamic_growth_steps=None)
        self.inner_optimizer.gradient_transformers.insert(
            0, lambda grads_and_vars: [(tf.cast(g, dtype=tf.float32) / loss_scaling, v) for g, v in grads_and_vars]
        )

    def get_unscaled_gradients(self, grads):
        # We do the gradient unscaling in the inner optimizer gradient
        # transforms instead of here, which uses less memory.
        return grads

    def _raise_if_strategy_unsupported(self):
        # This disallows the popdist distributed strategy to use
        # the static loss scale optimizer. If dynamic is False, this
        # can be skipped.
        pass


def get_optimizer(
    optimizer_name,
    num_replicas,
    learning_rate_schedule,
    use_outlining=False,
    loss_scaling=None,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-6,
    weight_decay_rate=0.0,
):
    """Constructs and returns a Keras optimizer
    Args:
        optimizer_name: A string representing the name of the
            required optimizer
        num_replicas: An integer representing the number of replicas. This
            will be used to scale down the gradients.
        learning_rate_schedule: A float or a Keras learning rate schedule
            object.
        use_outlining: A boolean, if true, the optimizer update will be
            outlined.
        loss_scaling: A float representing the fixed loss scaling. If None,
            the loss will not be scaled.
        beta1: A `float` value or a constant `float` tensor.
            The exponential decay rate for the 1st moment estimates.
        beta2: A `float` value or a constant `float` tensor.
            The exponential decay rate for the 2nd moment estimates.
        epsilon: A small constant for numerical stability.
        weight_decay_rate: A `float` value to decay the variables by in the
            gradient update step.
    """

    def rescale_gradients_down(grads_and_vars):
        return [(tf.cast(g, dtype=tf.float32) / num_replicas, v) for g, v in grads_and_vars]

    optimizer_kwargs = {
        "learning_rate": learning_rate_schedule,
        "beta_1": beta1,
        "beta_2": beta2,
        "epsilon": epsilon,
        "m_dtype": tf.float32,
        "v_dtype": tf.float32,
        "gradient_transformers": [rescale_gradients_down],
        "outline_apply_gradients": use_outlining,
        "outline_apply_gradients_kwargs": {"unique_sharding": True} if use_outlining else None,
    }

    if optimizer_name == "AdamW":
        AdamWIpuOptimizer = extend_with_decoupled_weight_decay(AdamIpuOptimizer)
        optimizer = AdamWIpuOptimizer(
            weight_decay_rate,
            optimizer_compute_precisions=(tf.float32,),
            **optimizer_kwargs,
        )
    elif optimizer_name == "LAMB":
        optimizer = LAMBIpuOptimizer(
            weight_decay_rate=weight_decay_rate,
            exclude_from_layer_adaptation=["bias", "beta", "gamma"],
            exclude_from_weight_decay=["bias", "beta"],
            optimizer_compute_precisions=(tf.float32, tf.float16),
            **optimizer_kwargs,
        )
    else:
        raise ValueError(f"Unrecognised optimizer name: `{optimizer_name}`." f" Choose one of {ALLOWED_OPTIMIZERS}")

    if loss_scaling is not None:
        optimizer = StaticLossScaleOptimizer(optimizer, loss_scaling)

    return optimizer
