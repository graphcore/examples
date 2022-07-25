# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import argparse
from tensorflow.python.ipu.config import IPUConfig
import numpy as np
import copy
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
import json
import logging
from logging import getLogger
import utils
from functools import reduce
from operator import add
import sys

tf.disable_eager_execution()
tf.disable_v2_behavior()

logging.basicConfig(
    level=logging.getLevelName("DEBUG"),
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = getLogger("bs)_matmul")

tf.logging.set_verbosity(tf.logging.ERROR)
np.set_printoptions(linewidth=200)


@tf.function
def log_loss(y, yHat):
    one = tf.constant(1.0, dtype=y.dtype)
    losses = y * tf.math.log(tf.clip_by_value(yHat, 0.000001, 1.0)) + (one - y) * tf.math.log(tf.clip_by_value(one - yHat, 0.000001, 1.0))
    return tf.reduce_sum(losses)


@tf.function
def log_loss_masked(y, yHat, mask):
    one = tf.constant(1.0, dtype=y.dtype)
    losses = y * tf.math.log(tf.clip_by_value(yHat, 0.000001, 1.0)) + (one - y) * tf.math.log(tf.clip_by_value(one - yHat, 0.000001, 1.0))
    losses = losses * mask
    return tf.reduce_sum(losses)


def bs_softmax_test(opts):
    dim = [opts.rows, opts.cols]
    block_size = [opts.block_row, opts.block_col]
    block_dim = [0] * 2
    for i in range(2):
        assert(dim[i] > 0)
        assert(block_size[i] > 0)
        assert(dim[i] % block_size[i] == 0)
        block_dim[i] = dim[i] // block_size[i]

    if opts.sparsity_mask is not None:
        sparsity_or_mask = list(int(c) for c in opts.sparsity_mask)
    else:
        sparsity_or_mask = opts.sparsity

    inner_group_size = opts.inner_group_size
    compute_grads = opts.compute_grads

    assert(isinstance(opts.subblock_mask_type, list))
    assert(len(opts.subblock_mask_type) > 0)
    subblock_dict = {"no": 0, "zut": 1, "zlt": 2}
    subblock_mask_type = list(subblock_dict[sm] for sm in opts.subblock_mask_type)
    if opts.group_dims is not None:
        if len(subblock_mask_type) == 1 and len(opts.group_dims) > 1:
            subblock_mask_type = subblock_mask_type * len(opts.group_dims)
    subblock_mask_flag = reduce(add, subblock_mask_type, 0)

    op_name = "BuildSoftmaxInPlace" if opts.in_place else "BuildSoftmax"

    dim_dense = copy.deepcopy(dim)
    dim_sparse_mask = [block_dim[0], block_dim[1]]
    if opts.group_dims is not None:
        dim_dense = opts.group_dims + dim_dense
        dim_sparse_mask = opts.group_dims + dim_sparse_mask

    sparse_logits, dense_masked_logits, sparsity_mask = utils.create_block_sparse_tensor(dim_dense, block_size, sparsity_or_mask, initial_value=-1000.0)

    nz = reduce(add, sparsity_mask, 0)
    logger.debug(f"sparsity_mask: {sparsity_mask}, nz blocks: {nz}")
    dim_block_sparse = [nz, block_size[0] * block_size[1]]

    diag_mask = None
    if subblock_mask_flag > 0:
        # Masked element = 0, unmasked = 1
        diag_mask = utils.create_diagonal_mask(dim_dense, subblock_mask_type)
        # Masked element = -1000, unmasked = 1000
        diag_mask_logits = diag_mask * 2000.0 - 1000.0

        # Masked elements become -1000
        # Unmmasked elements unchanged (assuming they are < 1000)
        dense_masked_logits = np.minimum(dense_masked_logits, diag_mask_logits)

    empty_rows_mask_np = utils.create_empty_rows_mask(dim_dense, sparsity_mask, block_size, diag_mask)
    empty_rows_mask_sparse_np = utils.to_block_sparse(empty_rows_mask_np, block_size, sparsity_mask)

    labels_ref_np = utils.create_random_sparse_labels(dim_dense, sparsity_mask, block_size)
    labels_np = utils.to_block_sparse(labels_ref_np, block_size, sparsity_mask, "int")

    tf_type = tf.float32 if opts.data_type == "float" else tf.float16

    logits = tf.Variable(sparse_logits, dtype=tf_type)
    logits_ref = tf.Variable(dense_masked_logits, dtype=tf_type)

    labels_ref = tf.Variable(labels_ref_np, dtype=tf_type)
    labels = tf.Variable(labels_np, dtype=tf_type)

    empty_rows_mask_ref = tf.Variable(empty_rows_mask_np, dtype=tf_type)
    empty_rows_mask = tf.Variable(empty_rows_mask_sparse_np, dtype=tf_type)

    if compute_grads:
        def dense_softmax(logits_ref):
            with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE, use_resource=True):
                probs = tf.nn.softmax(logits_ref)
                probs = probs * empty_rows_mask_ref
                loss = log_loss_masked(labels_ref, probs, empty_rows_mask_ref)
                logits_grad = tf.gradients(loss, logits_ref)
                return probs, loss, logits_grad
    else:
        def dense_softmax(logits_ref):
            with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE, use_resource=True):
                probs = tf.nn.softmax(logits_ref)
                probs = probs * empty_rows_mask_ref
                loss = log_loss_masked(labels_ref, probs, empty_rows_mask_ref)
                return probs, loss

    bs_softmax_args = {
        "dim_dense": dim_dense,
        "block_size": block_size,
        "sparsity_mask": "".join(str(c) for c in sparsity_mask),
        "subblock_mask_type": subblock_mask_type,
        "inner_group_size": inner_group_size
    }
    json_attribs = json.dumps(bs_softmax_args)

    logger.debug(f"json_attribs: {json_attribs}")

    if compute_grads:
        def bs_softmax(logits):
            outputs = {
                "output_types": [tf_type],
                "output_shapes": [tf.TensorShape(dim_block_sparse)]}
            lib_path = utils.get_lib_path("block_sparse")

            with tf.variable_scope("bs_softmax", reuse=tf.AUTO_REUSE, use_resource=True):
                probs = ipu.custom_ops.precompiled_user_op(
                    [logits],
                    lib_path,
                    outs=outputs,
                    op_name=op_name,
                    separate_gradients=False,
                    inputs_with_gradients=[0],
                    attributes=json_attribs,
                    gradient_attributes=json_attribs)

            loss = log_loss_masked(labels, probs, empty_rows_mask)
            logits_grad = tf.gradients(loss, logits)
            return probs, loss, logits_grad
    else:
        def bs_softmax(logits):
            outputs = {
                "output_types": [tf_type],
                "output_shapes": [tf.TensorShape(dim_block_sparse)]}
            lib_path = utils.get_lib_path("block_sparse")
            with tf.variable_scope("bs_softmax", reuse=tf.AUTO_REUSE, use_resource=True):
                probs = ipu.custom_ops.precompiled_user_op(
                    [logits],
                    lib_path,
                    outs=outputs,
                    op_name=op_name,
                    separate_gradients=False,
                    inputs_with_gradients=[],
                    attributes=json_attribs,
                    gradient_attributes=json_attribs)

            loss = log_loss_masked(labels, probs, empty_rows_mask)
            return probs, loss

    # Configure the IPU:
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        dense_softmax_fetches = ipu.ipu_compiler.compile(dense_softmax, [logits_ref])
        bs_softmax_fetches = ipu.ipu_compiler.compile(bs_softmax, [logits])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        results_ref = sess.run(dense_softmax_fetches)
        results = sess.run(bs_softmax_fetches)

    if compute_grads:
        probs_ref, loss_ref, logits_grad_ref = (results_ref[0], results_ref[1], results_ref[2][0])
        probs, loss, logits_grad = (results[0][0], results[1], results[2][0])
    else:
        probs_ref, loss_ref, logits_grad_ref = (results_ref[0], results_ref[1], None)
        probs, loss, logits_grad = (results[0], results[1], None)

    probs_ref = utils.to_block_sparse(np.array(probs_ref), block_size, sparsity_mask)
    if compute_grads:
        logits_grad_ref = utils.to_block_sparse(np.array(logits_grad_ref), block_size, sparsity_mask)

    return probs, loss, logits_grad, probs_ref, loss_ref, logits_grad_ref

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="block-sparse softmax test tool")
    parser.add_argument("--rows", type=int, required=True,
                        help="The number of rows")
    parser.add_argument("--cols", type=int, required=True,
                        help="The number of columns")
    parser.add_argument("--block-row", type=int, default=8,
                        help="The block size for the row")
    parser.add_argument("--block-col", type=int, default=8,
                        help="The block size for the column")
    parser.add_argument("--data-type", type=str,
                        choices=["float", "half"],
                        default="float",
                        help="Data type")
    parser.add_argument("--compute-grads", action='store_true',
                        help="Compute gradients")
    parser.add_argument("--in-place", action='store_true',
                        help="Do op in place")
    parser.add_argument("--group-dims", nargs='*', type=int,
                        help="The list of group dimensions")
    parser.add_argument("--sparsity", type=float, default=0.0,
                        help="Level of sparsity (ignored if sparsity mask is provided)")
    parser.add_argument("--sparsity-mask", type=str,
                        help="Sparsity mask")
    parser.add_argument("--subblock-mask-type", nargs='*', type=str,
                        choices=["no", "zut", "zlt"],
                        default=["no"],
                        help="Subblock mask type:\n"
                        "no - no subblock mask\n"
                        "zut - zero upper triangle\n"
                        "zlt - zero lower triangle\n"
                        "You can provide individual subblock mask types for different groups in a list"
                        )
    parser.add_argument("--inner-group-size", type=int, default=1,
                        help="The size of groupped softmax (0 if perform softmax in 1 group)")
    opts = parser.parse_args()

    probs, loss, logits_grad, probs_ref, loss_ref, logits_grad_ref = bs_softmax_test(opts)

    if opts.data_type == "float":
        rtol = 1e-04
        atol = 1e-06
    else:
        rtol = 1e-01
        atol = 1e-03

    if not np.allclose(probs_ref, probs, rtol=rtol, atol=atol, equal_nan=True):
        print("probs result does not match")
        print(f"Reference probs:\n{probs_ref}")
        print(f"Sparse probs:\n{probs}")
        diff = np.array(probs_ref) - np.array(probs)
        print(f"Difference:\n{diff}")
    else:
        print("probs result matches")

    if not np.allclose(loss_ref, loss, rtol=rtol, atol=atol, equal_nan=True):
        print("loss result does not match")
        print(f"Reference loss:\n{loss_ref}")
        print(f"Loss:\n{loss}")
        diff = loss_ref - loss
        print(f"Difference:\n{diff}")
    else:
        print("loss result matches")

    if opts.compute_grads:
        if not np.allclose(logits_grad_ref, logits_grad, rtol=rtol, atol=atol, equal_nan=True):
            print("logits_grad result does not match")
            print(f"Reference logits_grad:\n{logits_grad_ref}")
            print(f"Sparse logits_grad:\n{logits_grad}")
            diff = np.array(logits_grad_ref) - np.array(logits_grad)
            print(f"Difference:\n{diff}")
        else:
            print("logits_grad result matches")
