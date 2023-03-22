# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from ipu_tensorflow_addons.keras.optimizers import AdamIpuOptimizer
from tensorflow.keras.mixed_precision import LossScaleOptimizer


def get_optimizer(
    gradient_accumulation_steps_per_replica,
    num_replicas,
    learning_rate,
    optimizer_compute_precision,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-06,
    loss_scaling=None,
):
    """Constructs and returns a Keras optimizer
    Args:
        gradient_accumulation_steps_per_replica: An integer representing the
            number of gradient accumulations that have occurred. This will be
            used to scale down the gradients.
        num_replicas: An integer representing the number of replicas. This
            will be used to scale down the gradients.
        learning_rate: A float representing a fixed learning rate.
        optimizer_compute_precision: A flag used to set the optimizer compute precision.
        beta_1: A float value representing the exponential decay rate for the 1st moment estimates.
        beta_2: A float value representing the exponential decay rate for the 2nd moment estimates.
        epsilon: A small constant for numerical stability.
        loss_scaling: A float representing the fixed loss scaling. If None,
            the loss will not be scaled.
    """

    def scale_gradients_by_replicas_and_grad_accum(grads_and_vars):
        # The accumulation of gradients and the combining of gradients
        # over replicas are both implemented as a sum by default. We
        # need to normalise both by dividing by the gradient accumulation
        # steps and the number of replicas.
        scale = gradient_accumulation_steps_per_replica * num_replicas
        return [(tf.cast(g, dtype=optimizer_compute_precision) / scale, v) for g, v in grads_and_vars]

    optimizer = AdamIpuOptimizer(
        optimizer_compute_precisions=(optimizer_compute_precision,),
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        m_dtype=optimizer_compute_precision,
        v_dtype=optimizer_compute_precision,
        gradient_transformers=[scale_gradients_by_replicas_and_grad_accum],
    )

    if loss_scaling is not None:
        optimizer = StaticLossScaleOptimizer(optimizer, loss_scaling, optimizer_compute_precision)

    return optimizer


class StaticLossScaleOptimizer(LossScaleOptimizer):
    """
    Modification of the Keras LossScaleOptimizer that only
    supports a static loss scaling to use mixed precision in
    pipelining.
    """

    def __init__(self, inner_optimizer, loss_scaling, optimizer_compute_precision):
        """
        Constructs loss scale optimizer.
        :param inner_optimizer: The optimizer to wrap with this
        functionality.
        :param loss_scaling: Constant loss scaling factor to be applied.
        """
        super().__init__(inner_optimizer, dynamic=False, initial_scale=loss_scaling)
        self.inner_optimizer.gradient_transformers.insert(
            0,
            lambda grads_and_vars: [
                (tf.cast(g, dtype=optimizer_compute_precision) / loss_scaling, v) for g, v in grads_and_vars
            ],
        )

    def get_unscaled_gradients(self, grads):
        # We do the gradient unscaling in the inner optimizer gradient
        # transforms instead of here, which uses less memory.
        return grads
