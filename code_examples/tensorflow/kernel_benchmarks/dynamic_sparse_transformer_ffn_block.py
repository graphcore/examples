#!/usr/bin/env python
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
Benchmark the ffn block inside a dynamic sparse transformer
"""
import os
import inspect
import logging
import numpy as np
from functools import partial

import tensorflow.compat.v1 as tf
from tensorflow.python.ipu import utils, scopes, ipu_outfeed_queue
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import ipu_compiler

cwd = os.path.dirname(os.path.abspath(__file__))
os.sys.path.insert(1, os.path.join(cwd, '..', '..', '..', 'applications', 'tensorflow', 'dynamic_sparsity'))
from ipu_sparse_ops import sparse, layers, optimizers  # noqa: E402
from ipu_sparse_ops.model_baseclass import SparseModelOptions  # noqa: E402
from ipu_sparse_ops.transformer.transformer_baseclass import TransformerOptions   # noqa: E402
from ipu_sparse_ops.transformer.transformer_dynsparse import DynsparseTransformer  # noqa: E402

tf.disable_eager_execution()
tf.disable_v2_behavior()


def add_args(parser):
    TransformerOptions.add_all_arguments(parser)
    SparseModelOptions.add_all_arguments(parser)
    parser.add_argument("--mode", choices=['train', 'infer'], default="infer", help="mode choices: [train, infer]")
    parser.add_argument("--compute-dense-grad", default=False, help="If training, compute dense grads in backward pass")

    defaults = dict(
        dtype=tf.float32,
        batch_size=1,
        sparsity=0.9,
        source_sequence_length=256,
        hidden_length=1024,
        ff_length=4 * 1024,
        batches_per_step=5000,
        random_seed=11,
        disable_updating=True
    )
    parser.set_defaults(**defaults)
    return parser


def inputs(opts, index):
    value = tf.cast(index, opts.dtype)
    inputs = tf.broadcast_to(value, [opts.batch_size, opts.source_sequence_length, opts.hidden_length])
    return {"input_activation": inputs}


def graph_builder(opts, inputs):
    input_activation = inputs["input_activation"]
    transformer = DynsparseTransformer(opts)
    transformer.compute_dense_grad = opts.compute_dense_grad and opts.mode == 'train'
    output_activation = transformer.feed_forward(input_activation,
                                                 compute_dense_grad=opts.compute_dense_grad and
                                                 opts.mode == 'train')
    loss = tf.reduce_sum(output_activation)
    output = loss

    if opts.mode == 'train':
        with tf.variable_scope("train", reuse=tf.AUTO_REUSE, use_resource=True):
            global_step = tf.train.get_or_create_global_step()
            optimizer = optimizers.SparseOptimizer(tf.train.AdamOptimizer)
            optimizer = optimizer(learning_rate=1e-3, sparse_layers=transformer.sparse_layers.values())
            train_op = optimizer.minimize(loss, global_step=global_step)
            input_grad = tf.gradients(loss, input_activation)[0]

            dense_grads = []
            if opts.compute_dense_grad:
                dense_grads = list(transformer.streamDenseGradsFromDevice(loss, optimizer, {}).values())
            with tf.control_dependencies(dense_grads + [train_op, input_grad]):
                output = tf.identity(loss)

    return output


def get_ffn_flops(opts):
    B, S = opts.batch_size, opts.source_sequence_length
    H, F = opts.hidden_length, opts.ff_length
    forward_work = B * S * (H * 2) * F + B * S * (F * 2) * H
    backward_work = 2 * forward_work
    work = forward_work + backward_work * int(opts.mode == 'train')
    work *= (1 - opts.sparsity)
    return work


def iteration_report(opts, time):
    work = get_ffn_flops(opts)
    tflops_per_sec = opts.batches_per_step * opts.replicas * work * 1e-12 / time
    tokens_per_sec = opts.batch_size * opts.source_sequence_length * opts.batches_per_step * opts.replicas / time
    return f"{tokens_per_sec:.1f} tokens/sec, problem size {work*1e-9:.1f} GFLOPS, {tflops_per_sec:.3f} TFLOPS/sec"


def out_shape(builder, opts, inputs):
    with ipu_scope('/device:IPU:0'):
        return ipu_compiler.compile(partial(builder, opts), [inputs])[0]


if __name__ == '__main__':
    # Add benchmark module to path
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    os.sys.path.insert(1, os.path.join(cwd, '..', '..', '..', 'utils', 'benchmarks', 'tensorflow'))
    import benchmark  # noqa: E402
    logging.basicConfig(
        level=logging.getLevelName("DEBUG"),
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    module = benchmark.Benchmark(
        graph_builder,
        inputs,
        tf.global_variables_initializer,
        add_args,
        iteration_report,
        None,
        out_shape
    )

    opts = benchmark.parse_opts(module, False)
    np.random.seed(opts.random_seed)

    if opts.shards > 1:
        raise NotImplementedError("--shards option has not been implemented with this example")
    if opts.replicas > 1:
        raise NotImplementedError("--replicas option has not been implemented with this example")

    print(f" Dynamic Sparse Transformer Feed-forward Layer {'Train' if opts.mode == 'train' else 'Inference'} Synthetic benchmark.\n"
          f" Batch size {opts.batch_size}.\n"
          f" Batches per Step {opts.batches_per_step if not opts.report else 'n/a'}.\n"
          f" Sequence length {opts.source_sequence_length}\n"
          f" Hidden length {opts.hidden_length}\n"
          f" Ff length {opts.ff_length}.\n"
          f" Sparsity {opts.sparsity}.\n")
    benchmark.run(module, opts)
