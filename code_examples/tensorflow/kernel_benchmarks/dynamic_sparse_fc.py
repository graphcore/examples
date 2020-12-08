#!/usr/bin/env python
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
Benchmark a single dynamic sparse FC layer.

The figures reported at the end of the benchmark are based on wall time.

Run with -h or --help for options.
"""
import inspect
import os
import sys
import tensorflow.compat.v1 as tf
import numpy as np
from functools import partial
import logging
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cwd, '..', '..', '..', 'applications',
                                'tensorflow', 'dynamic_sparsity'))
from ipu_sparse_ops import sparse, layers, optimizers  # noqa
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import ipu_compiler

tf.disable_eager_execution()
tf.disable_v2_behavior()


def add_args(parser):
    parser.add_argument("--batch-size", type=int,
                        help="Number of vectors in a mini-batch")
    parser.add_argument("--input-size", type=int,
                        help="Input size (rows of sparse weight matrix)")
    parser.add_argument("--output-size", type=int,
                        help="Output size (cols of sparse weight matrix)")
    parser.add_argument("--density", type=float,
                        help="Non-zero density (e.g. 0.1 => 10 percent of elements are non-zero).")
    parser.add_argument("--block-size", type=int,
                        help="Size of square non-zero blocks.",
                        choices=[1, 4, 8, 16])
    parser.add_argument("--np-seed", type=int,
                        help="Set the random seed for numpy.")
    parser.add_argument("--mode", choices=['infer', 'train'], default='infer',
                        help="mode to run, train will compute loss and optimization pass")
    parser.add_argument('--meta-info-oversize', type=float,
                        help="Sets the Popsparse matmul option 'metaInfoBucketOversizeProportion'.")
    parser.add_argument("--available-memo-proportion", type=float,
                        help="Sets the available memory proportion")
    parser.add_argument("--data-type", type=str,
                        help="Choose the floating point type for the weights.",
                        choices=['fp32', 'fp16'])
    parser.add_argument("--partials-type", type=str,
                        help="Choose the floating point type for the weights.",
                        choices=['fp32', 'fp16'])
    parser.set_defaults(block_size=1, batch_size=32, input_size=1024, output_size=1024,
                        density=0.1, np_seed=42, batches_per_step=5000, steps=5,
                        meta_info_oversize=0.2, data_type='fp32', partials_type='fp32',
                        available_memo_proportion=1.0)

    return parser


def get_matmul_options(args):
    do_grads = "true" if args.mode == 'train' else "false"
    partialsType = "half" if args.partials_type == "fp16" else "float"
    matmul_options = {
        "availableMemoryProportion": args.available_memo_proportion,
        "doGradAPass": do_grads,
        "doGradWPass": do_grads,
        "metaInfoBucketOversizeProportion": args.meta_info_oversize,
        "partialsType": partialsType
    }
    return matmul_options


def make_random_sparse_fc_layer(args):
    random_gen = np.random.default_rng(seed=args.np_seed)
    weights_type = tf.float16 if args.data_type == 'fp16' else tf.float32

    fc = layers.SparseFcLayer.from_random_generator(
        args.output_size, [args.batch_size, args.input_size],
        args.density, args.block_size,
        random_gen.standard_normal, random_gen,
        matmul_options=get_matmul_options(args),
        name='sparse_fc_benchmark',
        use_bias=False, relu=False, disable_updating=not args.use_generated_data,
        dtype=weights_type)

    return fc


def inputs(opts, index):
    weights_type = tf.float16 if opts.data_type == 'fp16' else tf.float32
    value = tf.cast(index, weights_type)
    inputs = tf.broadcast_to(value, [opts.batch_size, opts.input_size])
    return {
        "inputs": inputs,
        "cond": tf.constant(False)
    }


def graph_builder(layers, opts, inputs):
    layers['fc'] = layers['fc_gen']()

    if opts.mode == 'train':
        # Need to check if this is the last iteration of the loop
        with tf.variable_scope("counter", reuse=tf.AUTO_REUSE, use_resource=True):
            itr_counter = tf.get_variable("iterations", shape=[], dtype=tf.int32,
                                          initializer=tf.zeros_initializer())
            mod_itrs = tf.math.floormod(itr_counter, opts.batches_per_step)
            last_itr = tf.equal(mod_itrs, 0)
            inc = tf.assign_add(itr_counter, 1)
        z = layers['fc'](inputs["inputs"], last_itr)
        # Loss is the mean across output matrix:
        loss = tf.reduce_mean(z)
        with tf.variable_scope("train", reuse=tf.AUTO_REUSE, use_resource=True):
            # We need to ensure that the train op is executed as part of
            # the benchmarking loop by maintaining a step variable and
            # forcing a control dependency between it and the train op:
            global_step = tf.get_variable(
                "step_control", dtype=tf.int32, shape=[])
            optimiser = optimizers.SparseOptimizer(tf.train.GradientDescentOptimizer)(
                learning_rate=0.01, name='optimise',
                sparse_layers=[layers['fc']])
            with tf.control_dependencies([global_step]):
                train_op = optimiser.minimize(loss)
        grad_input = tf.gradients(loss, inputs["inputs"])[0]
        all_ops = tf.group(inc, train_op, grad_input)
        with tf.control_dependencies([all_ops]):
            global_step = tf.identity(global_step)
        return global_step

    else:
        return layers['fc'](inputs["inputs"], inputs["cond"])


def initializer_sess(layers, utils, sess):
    with tf.device("cpu"):
        layers["fc"].create_placeholders()


def initializer():
    return tf.global_variables_initializer()


def out_shape(builder, opts, inputs):
    with ipu_scope('/device:IPU:0'):
        return ipu_compiler.compile(partial(builder, opts), [inputs])[0]


def iteration_report(opts, time):
    fwd_work_tflops = 1e-12 * 2 * opts.batch_size * opts.input_size * opts.output_size * opts.density
    bwd_work_tflops = 2 * fwd_work_tflops
    tflops_per_batch = fwd_work_tflops + int(opts.mode == 'train') * bwd_work_tflops
    items_per_sec = opts.batch_size * opts.batches_per_step * opts.replicas / time
    tflops_per_sec = tflops_per_batch * opts.batches_per_step * opts.replicas / time
    return f"{items_per_sec:5f} items/sec {tflops_per_sec:.2f} TFLOPS/sec"


if __name__ == '__main__':
    # Add benchmark module to path
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    sys.path.insert(1, os.path.join(cwd, '..', '..', '..', 'utils',
                                    'benchmarks', 'tensorflow'))
    import benchmark

    logging.basicConfig(
        level=logging.getLevelName("DEBUG"),
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    layer_dict = {}
    utils = {"train": False}

    module = benchmark.Benchmark(
        partial(graph_builder, layer_dict),
        inputs,
        initializer,
        add_args,
        iteration_report,
        partial(initializer_sess, layer_dict, utils),
        out_shape
    )

    # Make the sparse FC layer and insert into the layer dictionary
    # for use by the builder functions:
    options = benchmark.parse_opts(module, False)
    np.random.seed(options.np_seed)
    layer_dict['fc_gen'] = lambda: make_random_sparse_fc_layer(options)

    if options.mode == 'train':
        utils["train"] = True

    fc = layer_dict['fc_gen']()
    print(f"Metainfo bytes: {fc.weights.representation.metainfo_state.size * 2}")
    print(f"Non-zeros bytes: {fc.weights.representation.nz_values.size * 4}")

    if options.shards > 1:
        raise NotImplementedError(
            "--shards option has not been implemented with this example")

    if options.replicas > 1:
        raise NotImplementedError(
            "--replicas option has not been implemented with this example")

    # Log Benchmark Message
    print(" Dynamic Sparse FC Layer {} Synthetic benchmark.\n"
          " Batch size {}.\n"
          " Batches per Step {}.\n"
          " Equivalent dense size {}x{}.\n"
          f" Density {options.density} (sparsity {1-options.density}).\n"
          .format(
              "Inference",
              options.batch_size,
              options.batches_per_step if not options.report else "n/a",
              options.input_size, options.output_size))

    benchmark.run(module, options)
