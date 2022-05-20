# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import json
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu


def make_matmul_json_args(lhs, rhs, matmul_options: dict) -> str:
    """
    Utility function to create the attribute string expected by the
    sharded.matmul custom-op implementation.

    :param lhs: Tensor left-hand-side input to matmul (only used to get shape information).
    :param rhs: Tensor right-hand-side input to matmul (only used to get shape information).
    :param matmul_options: Dict mapping str -> str of Poplar matmul options.
    :return: JSON string of options that can be passed as a customop attribute argument.
    """
    # These attributes have to match those expected
    # by the reader function in utils.cpp:
    args = {
      "batch_size": int(lhs.shape[0]),
      "input_size": int(lhs.shape[1]),
      "output_size": int(rhs.shape[1]),
      "matmul_options": json.dumps(matmul_options)  # Poplar can parse this JSON string directly.
    }
    return json.dumps(args)


def make_gather_json_args(feature_count, feature_dim, index_count: int, grad_scale: float, slice_options: dict) -> str:
    """
    Utility function to create the attribute string expected by the
    sharded.embedding custom-op implementation.

    :param feature_count: Number of features in the lookup (rows of embedding matrix).
    :param feature dim: Dimension of each feature in the lookup (cols of embedding matrix).
    :index_count: The number of lookups (number of gathered rows).
    :grad_scale: Optional scale factor applied to the gradient.
    :param slice_options: Dict mapping str -> str containing Poplar slice options.
    :return: JSON string of options that can be passed as a customop attribute argument.
    """
    # The following attributes have to match those expected by the
    # reader function in utils.cpp:
    args = {
      "feature_count": int(feature_count),
      "feature_dim": int(feature_dim),
      "output_count": int(index_count),
      "gradient_scale": int(grad_scale),
      "slice_options": json.dumps(slice_options)
    }
    return json.dumps(args)


def matmul(lhs, rhs, matmul_options: dict, name: str = "sharded.matmul"):
    """
    Perform a sharded matmul where the right hand side (typically weights) has
    been split across shards by rows and the left hand side (typically the
    input activations) is replicated on every shard.

    :param lhs: Left hand side input tensor of shape (B, J, K)
    :param rhs: Right hand side input tensor of shape (K, M)
    :param matmul_options: Dict mapping str -> str of Poplar matmul options.
    :param name: Debug name for the op.
    :return: Matrix multiply result of shape (B, J, M)
    """
    # LHS is broadcast across IPUs (so outer dimension is the IPU index):
    json_args = make_matmul_json_args(lhs[0], rhs, matmul_options)
    batch_size = lhs.shape[1]
    output_size = rhs.shape[1]
    outputs = {
        "output_types": [lhs.dtype],
        "output_shapes": [tf.TensorShape([batch_size, output_size])],
    }
    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "libconcurrent_ops.so")
    return ipu.custom_ops.precompiled_user_op(
        [lhs, rhs], lib_path,
        outs=outputs,
        name=name,
        op_name='sharded_matmul',
        attributes=json_args,
        gradient_attributes=json_args,
        inputs_with_gradients=[0, 1],
        separate_gradients=False)[0]


def allocate_tied_embedding(tied_weights, indices, slice_options: dict, name: str = "sharded.allocate_tied_embedding"):
    """
    Allocator function for tensors inputs to sharded.embedding.

    The tensors used in shareded embedding need both special device placement and special tile
    layout within each for optimal performance. This op is functionally a no-op returning the
    same tensors but has the side effect of forcing the tensor to be allocated across devices
    efficiently.

    The the weight matrix to be sharded by row across all devices but the chunk within each
    device will have an embedding layout. The indices tensor will be allocated on a single
    device but with a tile layout used by Poplar for embedding indices. The indices tensor
    is not sharded as it is expected to be broadcast later (using sharded.to_all) in most
    applciations of embeddings.

    :param tied_weights: Tensor of tied embedding weights.
    :param indices: Tensor of tied embedding weights.
    :param slice_options: Dict mapping str -> str containing Poplar slice options.
    :param name: Debug name for the op.
    :return: The weights and indices tensors that were input to this function.
    """
    num_features = tied_weights.shape[0]
    feature_dim = tied_weights.shape[1]
    num_indices = indices.shape[0]
    grad_scale = 1.0
    json_args = make_gather_json_args(
      num_features, feature_dim, num_indices,
      grad_scale, slice_options)

    outputs = {
      "output_types": [tied_weights.dtype, indices.dtype],
      "output_shapes": [tied_weights.shape, indices.shape],
    }
    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "libconcurrent_ops.so")
    return ipu.custom_ops.precompiled_user_op(
        [tied_weights, indices], lib_path,
        outs=outputs,
        name=name,
        op_name='allocate_tied_embedding',
        attributes=json_args,
        inputs_with_gradients=[0, 1],
        separate_gradients=False)


def embedding(features, indices, slice_options: dict,
              name: str = "sharded.embedding"):
    """
    Sharded embedding lookup.

    :param features: Input embedding matrix of shape (R, C) (rows of this matrix will be gathered).
    :param indices: Tensor of shape (N, 1) containing indices of rows to gather.
    :param slice_options: Dict mapping str -> str containing Poplar slice options.
    :param name: Debug name for the op.

    :return: Tensor of shape (N, C) containing the lookup results.
    """
    # Pass parameters as attributes to the op so we can
    # also access them in the custom allocator function:
    num_features = features.shape[0]
    feature_dim = features.shape[1]
    num_indices = indices[0].shape[0]
    grad_scale = 1.0
    json_args = make_gather_json_args(
      num_features, feature_dim, num_indices,
      grad_scale, slice_options)

    # We need to define the output types and shapes for every custom op:
    outputs = {
        "output_types": [features.dtype],
        "output_shapes": [tf.TensorShape([num_indices, feature_dim])],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "libconcurrent_ops.so")
    return ipu.custom_ops.precompiled_user_op(
        [features, indices], lib_path,
        outs=outputs,
        name=name,
        op_name='embedding',
        attributes=json_args,
        gradient_attributes=json_args,
        inputs_with_gradients=[0],
        separate_gradients=True)[0]


def log_softmax(logits, grad_matmul_options: dict = None, name: str = "sharded.log_softmax"):
    """
    Sharded log-softmax operation across the last acis of the tensor,
    you should prefer to use the fused log_softmax_cross_entropy where possible.

    :param logits: Input tensor to softmax (typically logits: output of classifier layer).
    :param grad_matmul_options: A dictionary of str to str mappings describing Poplar matmul options
                                for the matmul in the backwards pass.
    :return: Tensor result of same shape as logits.
    """
    grad_attr = ""
    if grad_matmul_options:
        grad_attr = json.dumps(grad_matmul_options)

    outputs = {
        "output_types": [logits.dtype],
        "output_shapes": [logits.shape],
    }
    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "libconcurrent_ops.so")

    return ipu.custom_ops.precompiled_user_op(
        [logits], lib_path,
        outs=outputs,
        name=name,
        op_name='sharded_log_softmax',
        gradient_attributes=grad_attr,
        inputs_with_gradients=[0],
        separate_gradients=False)[0]


def log_softmax_cross_entropy(logits, indices, name: str = "sharded.log_softmax_cross_entropy"):
    """
    Fused sharded-softmax-cross-entropy across the last axis of the tensor.
    As in regular TensorFlow the fused operation makes the backwards pass much more
    computationally efficient and numerically stable, but additionally in this case
    avoids excess communication between shards (so it is critical to use this fused
    op when possible).

    :param logits: Input tensor to softmax (typically logits: output of classifier layer).
    :param indices: Ground truth indices for the cross-entropy calculation.
    :return: Tensor of .
    """
    outputs = {
      "output_types": [logits.dtype, logits.dtype],
      "output_shapes": [[logits.shape[0], 1], logits.shape],
    }
    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "libconcurrent_ops.so")

    return ipu.custom_ops.precompiled_user_op(
        [logits, indices], lib_path,
        outs=outputs,
        name=name,
        op_name='sharded_log_softmax_cross_entropy',
        inputs_with_gradients=[0],
        separate_gradients=False)[0]


def take_last(x, indices, name: str = "sharded.take_last"):
    """
    Gather scalars from the last axis of a tensor (useful for cross entropy loss).

    :param x: Tensor of shape (B, ..., N) to take from.
    :param indices: Indices tensor of shape (B, 1) into the last axis of x (one per batch item).
                    Out of range indices are ignored.

    :return: Tensor of shape (batch-size, 1) of gathered scalars.
    """
    if not x.shape[0] == indices[0].shape[0]:
        err = f"Incorrect number of indices (got: {indices.shape[0]} expected: {x.shape[0]})"
        raise RuntimeError(err)

    outputs = {
      "output_types": [x.dtype],
      "output_shapes": [[x.shape[0], 1]],
    }
    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "libconcurrent_ops.so")
    return ipu.custom_ops.precompiled_user_op(
        [x, indices], lib_path,
        outs=outputs,
        name=name,
        op_name='sharded_take_last',
        inputs_with_gradients=[0],
        separate_gradients=False)[0]


def to_all(input, num_ipus, name: str = "copy_to_all_shards"):
    """
    Operation to copy of the input tensor across a number of IPUs/shards.
    Note: The backwards pass of this operation is a reduction (sum) to the
    IPU on which the tensor originally resided.

    :param input: The tensor to be copied/broadcast.
    :param num_ipus: The number of IPUs (shards) to copy the input to.
    :return: A tensor with num_ipu copies of the input tensor in the outer-most axis
             (but each copy of tensor will be mapped to a different IPU).
    """
    outputs = {
      "output_types": [input.dtype],
      "output_shapes": [[num_ipus, *input.shape]],
    }
    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "libconcurrent_ops.so")
    return ipu.custom_ops.precompiled_user_op(
        [input], lib_path,
        outs=outputs,
        name=name,
        op_name='copy_to_all',
        inputs_with_gradients=[0],
        separate_gradients=False)[0]


def debug_op(input, name: str = "custom_debug"):
    """
    This is functionally a no-op but as a side effect
    it will print sharding information during graph construction.

    :param input: Tensor to print debug info about.
    :return: The input tensor.
    """
    outputs = {
      "output_types": [input.dtype],
      "output_shapes": [input.shape],
    }
    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "libconcurrent_ops.so")
    return ipu.custom_ops.precompiled_user_op(
        [input], lib_path,
        outs=outputs,
        name=name,
        op_name='debug',
        inputs_with_gradients=[0],
        separate_gradients=False)[0]
