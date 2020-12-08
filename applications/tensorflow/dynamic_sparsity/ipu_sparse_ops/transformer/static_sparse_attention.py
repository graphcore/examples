# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import json
import subprocess
import numpy as np
from functools import partial
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
from tensorflow.python.ipu import custom_ops
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu import ipu_compiler, scopes, loops, ipu_infeed_queue, ipu_outfeed_queue
tf.disable_eager_execution()
tf.disable_v2_behavior()


def get_lib_path(lib_name):
    base_path = os.path.realpath(os.path.dirname(__file__))
    return os.path.join(base_path, "..", "lib" + lib_name + ".so")


def dense_by_sparse_to_dense(d, s, sparsity_mask, blocksize2D, **kwargs):

    # The matmul shape (m, n) @ (n, k)
    *batch_dims, m, n = d.shape.with_rank_at_least(2).as_list()
    assert isinstance(sparsity_mask, list), "Sparsity mask should be a flat list of 0s and 1s"
    blocks_per_inner_group = len(sparsity_mask) // np.prod(batch_dims)   # e.g. [B, A, S, H] there are B*A "inner groups"
    blocks_in_dim_n = n // blocksize2D[0]
    blocks_in_dim_k = blocks_per_inner_group // blocks_in_dim_n
    k = int(blocks_in_dim_k * blocksize2D[1])

    # Data-type string has to be float or half
    data_type = "half" if d.dtype == tf.float16 else "float"

    # The defaults are set here, but can be overridden through kwargs
    # for instance the partial_data_type can be overried to float if desired
    bs_matmul_args = {
        "dim": [m, n, k],
        "block_size": [min(128, m)] + blocksize2D,
        "sparsity_mask": "".join(str(c) for c in sparsity_mask),
        "transposed_rhs": False,
        "data_type": data_type,
        "partial_data_type": data_type,
        "inner_group_size": int(np.prod(batch_dims)),  # how many of the batch dims to run in parallel
        "partition_method": "strip",
        "memory_cycle_ratio": 1
    }
    bs_matmul_args.update({k: v for k, v in kwargs.items() if k in bs_matmul_args})
    json_attribs = json.dumps(bs_matmul_args)

    # Call the custom operator which performs
    # [dense x sparse -> dense] matmul with
    # a static block-level sparsity mask
    y = custom_ops.precompiled_user_op(
        [d, s],
        get_lib_path("static_block_sparse"),
        outs={"output_types": [d.dtype], "output_shapes": [tf.TensorShape(list(batch_dims) + [m, k])]},
        op_name="BuildDSD",
        separate_gradients=False,
        inputs_with_gradients=[0, 1],
        attributes=json_attribs,
        gradient_attributes=json_attribs)[0]
    return y


def dense_by_dense_to_sparse(d1, d2, sparsity_mask, blocksize2D, **kwargs):

    # The matmul shape (m, n) @ (n, k) -> [num_blocks, block_area]
    *batch_dims, m, n = d1.shape.with_rank_at_least(2).as_list()
    k = d2.shape.with_rank_at_least(2).as_list()[-1]
    num_blocks = sum(sparsity_mask)
    block_area = np.prod(blocksize2D)

    # Data-type string has to be float or half
    data_type = "half" if d1.dtype == tf.float16 else "float"

    # The defaults are set here, but can be overridden through kwargs
    # for instance the partial_data_type can be overried to float if desired
    bs_matmul_args = {
        "dim": [m, n, k],
        "block_size": [blocksize2D[0], min(128, n), blocksize2D[1]],
        "sparsity_mask": "".join(str(c) for c in sparsity_mask),
        "transposed_rhs": False,
        "data_type": data_type,
        "partial_data_type": data_type,
        "inner_group_size": int(np.prod(batch_dims)),  # how many of the batch dims to run in parallel
        "partition_method": "strip",
        "memory_cycle_ratio": 1
    }
    bs_matmul_args.update({k: v for k, v in kwargs.items() if k in bs_matmul_args})
    json_attribs = json.dumps(bs_matmul_args)

    # Call the custom operator which performs
    # [dense x dense -> sparse] matmul with
    # a static block-level sparsity mask
    y = custom_ops.precompiled_user_op(
        [d1, d2],
        get_lib_path("static_block_sparse"),
        outs={"output_types": [d1.dtype], "output_shapes": [tf.TensorShape([num_blocks, block_area])]},
        op_name="BuildDDS",
        separate_gradients=False,
        inputs_with_gradients=[0, 1],
        attributes=json_attribs,
        gradient_attributes=json_attribs)[0]
    return y


def block_sparse_softmax(logits, dense_shape, sparsity_mask, blocksize2D, in_place=True, **kwargs):
    *batch_dims, m, n = dense_shape
    bs_softmax_args = {
        "dim_dense": dense_shape,
        "block_size": blocksize2D,
        "sparsity_mask": "".join(str(c) for c in sparsity_mask),
        "subblock_mask_type": [0] * int(np.prod(batch_dims)),  # 0: no mask, 1: zero upper triangle, 2: zero lower triangle
        "inner_group_size": int(np.prod(batch_dims))
    }
    bs_softmax_args.update({k: v for k, v in kwargs.items() if k in bs_softmax_args})
    json_attribs = json.dumps(bs_softmax_args)

    # Call the custom op, with optional in-placing
    op_name = "BuildSoftmaxInPlace" if in_place else "BuildSoftmax"
    probs = ipu.custom_ops.precompiled_user_op(
        [logits],
        get_lib_path("static_block_sparse"),
        outs={"output_types": [logits.dtype], "output_shapes": [tf.TensorShape(logits.shape)]},
        op_name=op_name,
        separate_gradients=False,
        inputs_with_gradients=[0],
        attributes=json_attribs,
        gradient_attributes=json_attribs)[0]
    return probs


def autoregressive_self_attention(q, kt, v, blocksize2D=[16, 16]):
    B, A, S, qkv_length = q.shape.with_rank(4).as_list()

    # Create block-autoregressive mask for all B*A batch dims
    mask = np.tril(np.ones([S // blocksize2D[0], S // blocksize2D[1]], np.int32), k=0)
    sparsity_mask = mask.flatten().tolist() * B * A

    # Interaction
    s = dense_by_dense_to_sparse(q, kt, sparsity_mask, blocksize2D)

    # Scale
    c = tf.constant(1 / np.sqrt(qkv_length), s.dtype)
    s = tf.multiply(s, c)

    # Softmax on block-sparse matrix
    # we specify the autoregressive "zero-upper-triangle" mask i.e. 1
    # FIXME change inner_group_size to B*A
    s = block_sparse_softmax(s, [B, A, S, S], sparsity_mask, blocksize2D, subblock_mask_type=[1] * B * A, inner_group_size=1)

    # Pick-up the dense values (the API presumes the sparse matrix is always
    # on the right side in the matmul: z=(v.T @ s.T).T
    # to transpose s we set the transpose_rhs flag to True
    vt = tf.transpose(v, [0, 1, 3, 2])
    zt = dense_by_sparse_to_dense(vt, s, sparsity_mask, blocksize2D, transposed_rhs=True)
    z = tf.transpose(zt, [0, 1, 3, 2])
    return z


def _dense_autoregressive_self_attention(q, kt, v):
    # Used for dev/testing
    B, A, S, qkv_length = q.shape.with_rank(4).as_list()

    # Interaction
    x = q @ kt

    # Scale
    c = tf.constant(1 / np.sqrt(qkv_length), x.dtype)
    x = tf.multiply(x, c)

    # Autoregressive mask and softmax
    mask = tf.constant(np.triu(np.ones([S, S]), k=1) * -10000, dtype=x.dtype)
    x = tf.add(x, mask, name="attention_mask")
    x = tf.nn.softmax(x, axis=-1)

    # Pickup the dense values
    z = x @ v
    return z


if __name__ == "__main__":
    # Run an example
    B = 1
    S = 64
    A = 4
    H = 256

    with tf.device("cpu"):
        q = tf.placeholder(tf.float32, shape=[B, A, S, H // A])
        kt = tf.placeholder(tf.float32, shape=[B, A, H // A, S])
        v = tf.placeholder(tf.float32, shape=[B, A, S, H // A])

    cfg = ipu.utils.auto_select_ipus(ipu.utils.create_ipu_config(), 1)
    ipu.utils.configure_ipu_system(cfg)
    with ipu.scopes.ipu_scope("/device:IPU:0"):
        sparse_out = ipu.ipu_compiler.compile(autoregressive_self_attention, [q, kt, v])
        dense_out = ipu.ipu_compiler.compile(_dense_autoregressive_self_attention, [q, kt, v])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.random.seed(0)
        feed_dict = {
            q: np.random.randn(B, A, S, H // A).astype(np.float32),
            kt: np.random.randn(B, A, H // A, S).astype(np.float32),
            v: np.random.randn(B, A, S, H // A).astype(np.float32)
        }
        sparse_result = sess.run(sparse_out, feed_dict)
        print("Sparse: ", sparse_result[0].sum())
        dense_result = sess.run(dense_out, feed_dict)
        print("Dense: ", dense_result[0].sum())

        s = sparse_result[0]
        d = dense_result[0]
        np.testing.assert_allclose(d, s, rtol=1e-05, atol=1e-05)
        print("Dense and sparse results are equal.")
