# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import jax
import jax.numpy as jnp
import numpy as np

"""
JAX functions to compute results and grads of various options
for use as a reference in testing.
"""


def gather(features, indices):
    """
    Performs a gather using a one-hot matmul.
    """
    one_hot = jax.nn.one_hot(indices, features.shape[0])
    output = jnp.dot(one_hot, features)
    reduced = jnp.sum(output)
    return output, reduced


def gather_grad_fn():
    """
    Return a function that computes the gradient of a gather.
    """
    def reduction_output(features, indices):
        return gather(features, indices)[1]
    return jax.grad(reduction_output, 0)


def matmul(a, b):
    output = jnp.matmul(a, b)
    reduced = jnp.sum(output)
    return output, reduced


def matmul_grad_fns():
    """
    Return a function to cmopute grads of matmul.
    """
    def reduction_output(a, b):
        return matmul(a, b)[1]
    return jax.grad(reduction_output, 0), jax.grad(reduction_output, 1)


def softmax_cross_entropy(logits, labels):
    nlsm = -jax.nn.log_softmax(logits)
    return jnp.take_along_axis(nlsm.squeeze(), labels, axis=-1)


def jax_loss_fn(logits, labels):
    ce = softmax_cross_entropy(logits, labels)
    loss = jnp.mean(ce)
    return loss


def projection_softmax_cross_entropy(a, b, labels):
    logits = jnp.matmul(a, b)
    loss = jax_loss_fn(logits, labels)
    return loss, logits


def projection_softmax_cross_entropy_grad_fns():
    def loss_only(a, b, labels):
        return projection_softmax_cross_entropy(a, b, labels)[0]
    # Return dL/da and dL/db:
    return jax.grad(loss_only, 0), jax.grad(loss_only, 1)
