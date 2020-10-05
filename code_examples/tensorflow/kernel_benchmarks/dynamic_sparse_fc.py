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
    parser.add_argument("--np-seed", type=int,
                        help="Set the random seed for numpy.")
    parser.add_argument("--train", action='store_true', dest='train',
                        help="Compute loss and optimization pass")
    parser.set_defaults(batch_size=32, input_size=1024, output_size=1024,
                        density=0.1, np_seed=42, batches_per_step=5000, steps=5)
    parser.add_argument('--meta-info-oversize', default=0.2, type=float,
                        help="Sets the Popsparse matmul option 'metaInfoBucketOversizeProportion'.")
    parser.add_argument("--data-type", type=str,
                        help="Choose the floating point type for the weights.",
                        choices=['fp32', 'fp16'])
    return parser


def get_matmul_options(args):
    do_grads = "true" if args.train else "false"
    matmul_options = {
      "availableMemoryProportion": 1.0,
      "doGradAPass": do_grads,
      "doGradWPass": do_grads,
      "metaInfoBucketOversizeProportion": args.meta_info_oversize
    }
    return matmul_options


def make_random_sparse_fc_layer(args):
    num_groups = 1
    topk_ratio = 0.5
    random_gen = np.random.default_rng(seed=args.np_seed)
    weights_type = tf.float16 if args.data_type == 'fp16' else tf.float32

    if args.train:
        fc = layers.SparseFcLayer.from_random_generator(
            args.output_size, [args.batch_size, args.input_size], args.density,
            random_gen.standard_normal, random_gen,
            matmul_options=get_matmul_options(args),
            name='sparse_fc_benchmark',
            bias=False, relu=False, disable_updating=not args.use_generated_data,
            dtype=weights_type)
    else:
        fc = layers.SparseFcLayer.from_random_generator(
            args.output_size, [args.batch_size, args.input_size], args.density,
            random_gen.standard_normal, random_gen,
            matmul_options=get_matmul_options(args),
            name='sparse_fc_benchmark',
            bias=False, relu=False, disable_updating=not args.use_generated_data,
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

    if opts.train:
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
            optimiser = optimizers.SparseOptimizer(tf.train.MomentumOptimizer)(
                learning_rate=0.01, momentum=0.0001, use_nesterov=True, name='optimise',
                sparse_layers=[layers['fc']])
            with tf.control_dependencies([global_step]):
                train_op = optimiser.minimize(loss)
        all_ops = tf.group(inc, train_op)
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


def iteration_report(opts, time):
    items_per_sec = opts.batch_size * opts.batches_per_step * opts.replicas / time
    tflops_per_sec = items_per_sec * 2 * opts.input_size * opts.output_size * opts.density * 1e-12
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
    )

    # Make the sparse FC layer and insert into the layer dictionary
    # for use by the builder functions:
    options = benchmark.parse_opts(module, False)
    np.random.seed(options.np_seed)
    layer_dict['fc_gen'] = lambda: make_random_sparse_fc_layer(options)

    if options.train:
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
