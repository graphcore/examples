# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from tensorflow_addons.optimizers.weight_decay_optimizers import extend_with_decoupled_weight_decay

from ipu_tensorflow_addons.keras.optimizers import AdamIpuOptimizer, LAMBIpuOptimizer


ALLOWED_OPTIMIZERS = ["AdamW", "LAMB"]


def get_optimizer(
    optimizer_name,
    gradient_accumulation_factor,
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
        gradient_accumulation_factor: An integer representing the number
            of gradient accumulations that have occured. This will be used
            to scale down the gradients.
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
        rescale_grads_by_factor = gradient_accumulation_factor * num_replicas
        return [
            (tf.cast(g, dtype=tf.float32) / rescale_grads_by_factor, v)
            for g, v in grads_and_vars
        ]

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
        AdamWIpuOptimizer = extend_with_decoupled_weight_decay(
            AdamIpuOptimizer)
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
        raise ValueError(
            f"Unrecognised optimizer name: `{optimizer_name}`."
            f" Choose one of {ALLOWED_OPTIMIZERS}")

    if loss_scaling is not None:
        optimizer = LossScaleOptimizer(optimizer,
                                       dynamic=False,
                                       initial_scale=loss_scaling)

    return optimizer
