# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
from tensorflow.python.ipu.config import IPUConfig
import argparse
import numpy as np
from functools import partial
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
from ipu_sparse_ops import sparse, layers, sparse_training
from tensorflow.python.ipu import ipu_outfeed_queue, ipu_compiler
from tensorflow.python.ipu.scopes import ipu_scope

tf.disable_eager_execution()
tf.disable_v2_behavior()


weights = {}


def get_program_arguments():
    parser = argparse.ArgumentParser(description='Sparse checkpoints tool')
    parser.add_argument("--input-size", type=int, default=16,
                        help="Input size of the layer.")
    parser.add_argument("--batchsize", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--output-size", type=int, default=16,
                        help="Output size of the layer.")
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32",
                        help="Floating point precision")
    parser.add_argument("--density", default=0.1, type=float, help="Density of the fc layer")
    parser.add_argument("--seed", default=0, type=int, help="numpy random seed")
    parser.add_argument("--print-weights", action='store_true',
                        help="Prints the dense fc weights at different "
                        "stages for debug purposes.")
    parser.add_argument('--pooling-type', default='SUM', choices=['SUM', 'AVG', 'MAX'],
                        help="Select dense gradients block pooling")
    parser.add_argument('--block-size', default=1, type=int, choices=[1, 4, 8, 16],
                        help="Sparse blocks size. Set to a value > 1 for block sparsity")
    parser.add_argument('--meta-info-oversize', default=0.1, type=float,
                        help="Sets the Popsparse matmul option 'metaInfoBucketOversizeProportion'.")
    return parser.parse_args()


def model(x_fc, fc, fc_pool, opts, outfeed_queue, dtype):
    with tf.variable_scope("SparseOps", reuse=tf.AUTO_REUSE, use_resource=True):
        y_fc = fc(x_fc, tf.constant(True))
        y_fc_pool = fc_pool(x_fc, tf.constant(True))
        loss = tf.reduce_sum(y_fc)
        loss_pool = tf.reduce_sum(y_fc_pool)
        output = {}
        output['dense_grad'] = tf.convert_to_tensor(fc.get_dense_grad_w(loss))
        output['pooled_dense_grad'] = tf.convert_to_tensor(fc_pool.get_dense_grad_w(loss_pool))
        out = outfeed_queue.enqueue(output)
        return out


def make_fc_weights(input_size, hidden_size, values):
    w = np.zeros([input_size, hidden_size])
    for value in values:
        w[np.random.randint(input_size), np.random.randint(hidden_size)] = value
    return w


def create_sparse_layers(opts):
    matmul_opts = {"metaInfoBucketOversizeProportion": opts.meta_info_oversize}
    in_blocks = opts.input_size // opts.block_size
    out_blocks = opts.output_size // opts.block_size
    identity_size = max(in_blocks, out_blocks)
    block_mask = np.identity(identity_size)[0: in_blocks, 0: out_blocks]
    block_mask[1, 3] = 1
    block_mask[0, 3] = 1
    n_blocks = np.count_nonzero(block_mask)
    el_mask = sparse.block_mask_to_element(block_mask, opts.block_size)
    n_els = np.count_nonzero(el_mask)
    masked_rhs = np.zeros_like(el_mask, dtype=np.float32 if opts.dtype == "fp32" else np.float16)
    values = np.random.rand(n_els)
    masked_rhs[np.nonzero(el_mask)] = values
    if opts.block_size == 1:
            triplets = sparse.triplets_from_dense(masked_rhs)
    else:
        triplets = sparse.triplets_from_dense(block_mask)
        triplets = sparse.Triplets(
            triplets.row_indices, triplets.col_indices,
            sparse.blocks_at_indices(
                triplets.row_indices, triplets.col_indices, opts.block_size, masked_rhs)
        )

    fc = layers.SparseFcLayer.from_triplets(
            opts.output_size, [opts.batchsize, opts.input_size], *triplets,
            matmul_options=matmul_opts,
            name="fc_None",
            dtype=dtype,
            use_bias=False, relu=False, pooling_type='NONE')
    fc_pool = layers.SparseFcLayer.from_triplets(
            opts.output_size, [opts.batchsize, opts.input_size], *triplets,
            matmul_options=matmul_opts,
            name="fc_" + opts.pooling_type,
            dtype=dtype,
            use_bias=False, relu=False, pooling_type=opts.pooling_type)

    return fc, fc_pool


def set_up_ipu_devices(opts):
    config = IPUConfig()
    config.auto_select_ipus = 1
    config.configure_ipu_system()
    # Set the seed for the stochastic rounding
    ipu.utils.reset_ipu_seed = opts.seed


def make_graph(fc_weights):
    graph = tf.Graph()

    with graph.as_default():
        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="sparse_outfeed")
        fc, fc_pool = create_sparse_layers(opts)

        model_op = partial(model, fc=fc, fc_pool=fc_pool,
                           opts=opts, outfeed_queue=outfeed_queue,
                           dtype=dtype)

        with tf.device("cpu"):
            x_fc = tf.placeholder(dtype, shape=[opts.batchsize, opts.input_size])

        with ipu_scope('/device:IPU:0'):
            test_op = ipu_compiler.compile(model_op, inputs=[x_fc])

        with tf.device("cpu"):
            fc.create_placeholders()
            fc_pool.create_placeholders()

        dequeue = outfeed_queue.dequeue()
        ipu.utils.move_variable_initialization_to_cpu()

    return graph, outfeed_queue, fc, fc_pool, x_fc, test_op, dequeue


if __name__ == "__main__":
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
    tmp_path = "./tmp/test"
    opts = get_program_arguments()
    set_up_ipu_devices(opts)

    dtype = tf.float32 if opts.dtype == 'fp32' else tf.float16

    np.random.seed(opts.seed)

    x_fc_in = np.random.normal(size=[opts.batchsize, opts.input_size])

    fc_weights = np.random.rand(10)

    # Create a first graph and run it to retrieve the weights from the ipu. Then create a checkpoint
    graph, outfeed_queue, fc, fc_pool, x_fc, test_op, dequeue = make_graph(fc_weights=fc_weights)

    with tf.Session(graph=graph) as sess:
        # init
        sess.run(tf.global_variables_initializer())

        # run and outfeed weights
        sess.run(test_op, feed_dict={x_fc: x_fc_in})
        results = sess.run(dequeue)

        unpooled = results['dense_grad'][0]
        ipu_pooled = results['pooled_dense_grad'][0]

        # do pooling on the host for the dense grad
        if opts.block_size == 1:
            cpu_pooled = unpooled
        else:
            cpu_pooled = sparse_training.block_pool(unpooled, opts.block_size, opts.pooling_type)

    if opts.dtype == 'fp16':
        atol = 1e-2
    else:
        atol = 1e-5
    if not np.allclose(cpu_pooled, ipu_pooled, atol=atol):
        raise Exception(f"Host and ipu pooling results don't match.\nHost pool:\n{cpu_pooled}\nIpu pool:\n{ipu_pooled}")

    print("All results match")
