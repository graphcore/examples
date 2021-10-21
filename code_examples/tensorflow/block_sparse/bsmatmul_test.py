# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import argparse
from tensorflow.python.ipu.config import IPUConfig
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
import logging
from logging import getLogger
import utils
import json
from functools import reduce
from operator import add

tf.disable_eager_execution()
tf.disable_v2_behavior()

logging.basicConfig(
    level=logging.getLevelName("DEBUG"),
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = getLogger("bs)_matmul")

tf.logging.set_verbosity(tf.logging.ERROR)
np.set_printoptions(linewidth=200)


def bs_matmul_test(opts):
    data_type = opts.data_type
    partial_data_type = opts.partial_data_type

    dim = [opts.lhs_rows, opts.lhs_cols, opts.rhs_cols]
    block_size = [opts.lhs_block_row, opts.lhs_block_col, opts.rhs_block_col]
    block_dim = [0] * 3
    for i in range(3):
        assert(dim[i] > 0)
        assert(block_size[i] > 0)
        assert(dim[i] % block_size[i] == 0)
        block_dim[i] = dim[i] // block_size[i]

    if opts.sparsity_mask is not None:
        sparsity_or_mask = list(int(c) for c in opts.sparsity_mask)
    else:
        sparsity_or_mask = opts.sparsity

    inner_group_size = opts.inner_group_size
    partition_method = opts.partition_method
    memory_cycle_ratio = opts.memory_cycle_ratio

    tf_type = tf.float32 if data_type == "float" else tf.float16

    sparse_out = False
    op_name = "BuildDSD"
    compute_grads = False
    if opts.scenario[:3] == "dds":
        sparse_out = True
        op_name = "BuildDDS"
    if len(opts.scenario) > 3:
        compute_grads = True

    transposed_rhs = False
    if (not sparse_out):
        transposed_rhs = opts.transposed_rhs

    if (not sparse_out):
        if not transposed_rhs:
            dim_dense = [dim[1], dim[2]]
            block_size_sparse = [block_size[1], block_size[2]]
            dim_sparse_mask = [block_dim[1], block_dim[2]]
        else:
            dim_dense = [dim[2], dim[1]]
            block_size_sparse = [block_size[2], block_size[1]]
            dim_sparse_mask = [block_dim[2], block_dim[1]]
    else:
        dim_dense = [dim[0], dim[2]]
        block_size_sparse = [block_size[0], block_size[2]]
        dim_sparse_mask = [block_dim[0], block_dim[2]]
    if opts.group_dims is not None:
        dim_dense = opts.group_dims + dim_dense
        dim_sparse_mask = opts.group_dims + dim_sparse_mask

    sparse_matrix, dense_masked_matrix, sparsity_mask = utils.create_block_sparse_tensor(dim_dense, block_size_sparse, sparsity_or_mask)
    if transposed_rhs:
        sparse_transposed_indices = list(range(len(dim_dense)))
        sparse_transposed_indices[-2], sparse_transposed_indices[-1] = sparse_transposed_indices[-1], sparse_transposed_indices[-2]
        dense_masked_matrix = dense_masked_matrix.transpose(sparse_transposed_indices)
        # leaving sparsity_mask is in transposed form

    nz = reduce(add, sparsity_mask, 0)
    logger.debug(f"sparsity_mask: {sparsity_mask}, nz blocks: {nz}")

    dim_lhs = [dim[0], dim[1]]
    dim_block_sparse = [nz, block_size_sparse[0] * block_size_sparse[1]]
    if opts.group_dims is not None:
        dim_lhs = opts.group_dims + dim_lhs
    if (not sparse_out):
        dim_rhs = dim_block_sparse
        dim_res = [dim[0], dim[2]]
        if opts.group_dims is not None:
            dim_res = opts.group_dims + dim_res
    else:
        dim_rhs = [dim[1], dim[2]]
        if opts.group_dims is not None:
            dim_rhs = opts.group_dims + dim_rhs
        dim_res = dim_block_sparse

    lhs_np = utils.create_dense_tensor(dim_lhs)
    lhs = tf.Variable(lhs_np, dtype=tf_type)

    if (not sparse_out):
        rhs = tf.Variable(sparse_matrix, dtype=tf_type)
        rhs_ref = tf.Variable(dense_masked_matrix, dtype=tf_type)
    else:
        rhs_np = utils.create_dense_tensor(dim_rhs)
        rhs = tf.Variable(rhs_np, dtype=tf_type)
        rhs_ref = rhs
        sparsity_mask_2d = np.reshape(sparsity_mask, dim_sparse_mask)
        block_one = np.ones([block_size[0], block_size[2]], dtype=np.float32)
        res_mask_np = np.kron(sparsity_mask_2d, block_one)
        res_mask = tf.constant(res_mask_np, dtype=tf_type)

    if (not sparse_out):
        if compute_grads:
            def dense_matmul(a, b):
                with tf.variable_scope("matmul", reuse=tf.AUTO_REUSE, use_resource=True):
                    c = tf.matmul(a, b)
                    s = tf.reduce_sum(c)
                    a_grad = tf.gradients(s, a)
                    b_grad = tf.gradients(s, b)
                    return c, a_grad, b_grad
        else:
            def dense_matmul(a, b):
                with tf.variable_scope("matmul", reuse=tf.AUTO_REUSE, use_resource=True):
                    c = tf.matmul(a, b)
                    return c
    else:
        if compute_grads:
            def dense_matmul(a, b):
                with tf.variable_scope("matmul", reuse=tf.AUTO_REUSE, use_resource=True):
                    c = tf.matmul(a, b)
                    c = c * res_mask
                    s = tf.reduce_sum(c)
                    a_grad = tf.gradients(s, a)
                    b_grad = tf.gradients(s, b)
                    return c, a_grad, b_grad
        else:
            def dense_matmul(a, b):
                with tf.variable_scope("matmul", reuse=tf.AUTO_REUSE, use_resource=True):
                    c = tf.matmul(a, b)
                    c = c * res_mask
                    return c

    bs_matmul_args = {
        "dim": dim,
        "block_size": block_size,
        "sparsity_mask": "".join(str(c) for c in sparsity_mask),
        "transposed_rhs": transposed_rhs,
        "data_type": data_type,
        "partial_data_type": partial_data_type,
        "inner_group_size": inner_group_size,
        "partition_method": partition_method,
        "memory_cycle_ratio": memory_cycle_ratio
    }
    json_attribs = json.dumps(bs_matmul_args)

    logger.debug(f"json_attribs: {json_attribs}")

    if compute_grads:
        def bs_matmul(a, b):
            outputs = {
                "output_types": [tf_type],
                "output_shapes": [tf.TensorShape(dim_res)]}
            lib_path = utils.get_lib_path("block_sparse")

            with tf.variable_scope("bs_matmul", reuse=tf.AUTO_REUSE, use_resource=True):
                c = ipu.custom_ops.precompiled_user_op(
                    [a, b],
                    lib_path,
                    outs=outputs,
                    op_name=op_name,
                    separate_gradients=False,
                    inputs_with_gradients=[0, 1],
                    attributes=json_attribs,
                    gradient_attributes=json_attribs)

                s = tf.reduce_sum(c)
                a_grad = tf.gradients(s, a)
                b_grad = tf.gradients(s, b)
            return c, a_grad, b_grad
    else:
        def bs_matmul(a, b):
            outputs = {
                "output_types": [tf_type],
                "output_shapes": [tf.TensorShape(dim_res)]}
            lib_path = utils.get_lib_path("block_sparse")

            with tf.variable_scope("bs_matmul", reuse=tf.AUTO_REUSE, use_resource=True):
                c = ipu.custom_ops.precompiled_user_op(
                    [a, b],
                    lib_path,
                    outs=outputs,
                    op_name=op_name,
                    separate_gradients=False,
                    inputs_with_gradients=[],
                    attributes=json_attribs)
            return c

    # Configure the IPU:
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        dense_matmul_fetches = ipu.ipu_compiler.compile(dense_matmul, [lhs, rhs_ref])
        bs_matmul_fetches = ipu.ipu_compiler.compile(bs_matmul, [lhs, rhs])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        results_ref = sess.run(dense_matmul_fetches)
        results = sess.run(bs_matmul_fetches)

    if compute_grads:
        out_ref, lhs_grad_ref, rhs_grad_ref = (results_ref[0], results_ref[1][0], results_ref[2][0])
        out, lhs_grad, rhs_grad = (results[0][0], results[1][0], results[2][0])
    else:
        out_ref, lhs_grad_ref, rhs_grad_ref = (results_ref[0], None, None)
        out, lhs_grad, rhs_grad = (results[0], None, None)

    if (sparse_out):
        out_ref = utils.to_block_sparse(np.array(out_ref), block_size_sparse, sparsity_mask)
    else:
        if compute_grads:
            rhs_grad_ref = np.array(rhs_grad_ref)
            if transposed_rhs:
                rhs_grad_ref = rhs_grad_ref.transpose(sparse_transposed_indices)
            rhs_grad_ref = utils.to_block_sparse(rhs_grad_ref, block_size_sparse, sparsity_mask)

    return out, lhs_grad, rhs_grad, out_ref, lhs_grad_ref, rhs_grad_ref

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="block-sparse matmul test tool")
    parser.add_argument("--scenario", type=str,
                        choices=["dsd", "dds", "dsd-grad", "dds-grad"],
                        default="dsd-grad",
                        help="Scenario:\n"
                             "dsd - test dense x sparse = dense\n"
                             "dsd-grad - test dense x sparse = dense and gradients\n"
                             "dds - test dense x dense = sparse\n"
                             "dsd-grad - test dense x sparse = dense and gradients\n"
                        )
    parser.add_argument("--data-type", type=str,
                        choices=["float", "half"],
                        default="float",
                        help="Data type")
    parser.add_argument("--partial-data-type", type=str,
                        choices=["float", "half"],
                        default="float",
                        help="Partials data type")
    parser.add_argument("--lhs-rows", type=int, required=True,
                        help="The number of rows for the LHS matrix")
    parser.add_argument("--lhs-cols", type=int, required=True,
                        help="The number of columns for the LHS matrix")
    parser.add_argument("--rhs-cols", type=int, required=True,
                        help="The number of columns for the RHS matrix")
    parser.add_argument("--lhs-block-row", type=int, default=8,
                        help="The block size for the row of rhe LHS matrix")
    parser.add_argument("--lhs-block-col", type=int, default=8,
                        help="The block size for the column of the LHS matrix")
    parser.add_argument("--rhs-block-col", type=int, default=8,
                        help="The block size for the column of the RHS matrix")
    parser.add_argument("--group-dims", nargs='*', type=int,
                        help="The list of group dimensions")
    parser.add_argument("--sparsity", type=float, default=0.0,
                        help="Level of sparsity (ignored if sparsity mask is provided)")
    parser.add_argument("--sparsity-mask", type=str,
                        help="Sparsity mask")
    parser.add_argument("--transposed-rhs", action='store_true',
                        help="RHS matrix transposed")
    parser.add_argument("--inner-group-size", type=int, default=1,
                        help="The size of groupped matmul (0 if perform matmul in 1 group)")
    parser.add_argument("--partition-method", type=str,
                        choices=["strip", "stripv0", "block-group2", "block-naive", "block"],
                        default="strip",
                        help="Partition method")
    parser.add_argument("--memory-cycle-ratio", type=float, default=1.0,
                        help="The ratio between memory weight and cycle weight (for 'block' partition method only)")
    opts = parser.parse_args()

    compute_grads = False
    if len(opts.scenario) > 3:
        compute_grads = True

    out, lhs_grad, rhs_grad, out_ref, lhs_grad_ref, rhs_grad_ref = bs_matmul_test(opts)

    rtol = 1e-05
    atol = 1e-06

    if not np.allclose(out_ref, out, rtol=rtol, atol=atol, equal_nan=True):
        print("out result does not match")
        print(f"Reference out:\n{out_ref}")
        print(f"Sparse out:\n{out}")
        diff = np.array(out_ref) - np.array(out)
        print(f"Difference:\n{diff}")
    else:
        print("out result matches")

    if compute_grads:
        if not np.allclose(lhs_grad_ref, lhs_grad, rtol=rtol, atol=atol, equal_nan=True):
            print("lhs_grad result does not match")
            print(f"Reference lhs_grad:\n{lhs_grad_ref}")
            print(f"Sparse lhs_grad:\n{lhs_grad}")
            diff = np.array(lhs_grad_ref) - np.array(lhs_grad)
            print(f"Difference:\n{diff}")
        else:
            print("lhs_grad result matches")

        if not np.allclose(rhs_grad_ref, rhs_grad, rtol=rtol, atol=atol, equal_nan=True):
            print("rhs_grad result does not match")
            print(f"Reference rhs_grad:\n{rhs_grad_ref}")
            print(f"Sparse rhs_grad:\n{rhs_grad}")
            diff = np.array(rhs_grad_ref) - np.array(rhs_grad)
            print(f"Difference:\n{diff}")
        else:
            print("rhs_grad result matches")
