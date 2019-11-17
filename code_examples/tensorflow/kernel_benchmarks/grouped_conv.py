#!/usr/bin/env python
# Copyright 2019 Graphcore Ltd.
"""
Benchmark a single CNN layer with no host/device data transfers.

The Items/sec reported at the end of the benchmark is based on wall time.

Run with -h or --help for options.
"""
import inspect
import os
import sys
import tensorflow as tf
from tensorflow.python.ipu import utils


def conv2d_basic(x, ksize, stride, filters_in, filters_out, index, groups=1,
                 name='conv'):
    with tf.variable_scope(name, use_resource=True):
        W = tf.get_variable("conv2d/kernel" + str(index),
                            shape=[ksize, ksize,
                                   filters_in / groups, filters_out],
                            dtype=x.dtype,
                            trainable=True,
                            initializer=tf.variance_scaling_initializer())
        return tf.nn.conv2d(x,
                            filters=W,
                            strides=[1, stride, stride, 1],
                            padding='SAME')


def conv(opts, inputs):
    groups = opts.filter_out // opts.group_dim
    results = inputs
    for i in range(opts.block_repeats):
        results = conv2d_basic(results, opts.kernel_size, opts.stride,
                               opts.filter_in, opts.filter_out, i,
                               groups=groups)
    return results


def inputs(opts, index):
    value = tf.cast(index, tf.float16)
    return {
        "inputs": tf.broadcast_to(value, [opts.batch_size,
                                          opts.input_size, opts.input_size,
                                          opts.filter_in]),
    }


def graph_builder(opts, inputs_dict):
    output = conv(opts, inputs_dict["inputs"])

    if opts.train:
        # dummy loss to minimize computational cost.
        loss = tf.reduce_mean(output)
        optimiser = tf.train.GradientDescentOptimizer(1e-30)
        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
            # We need to ensure that the train op is executed as part of
            # the benchmarking loop by maintaining a step variable and
            # forcing a control dependency between it and the train op:
            global_step = tf.get_variable(
                "step_control", dtype=tf.int32, shape=[])
            grads_and_vars = optimiser.compute_gradients(
                loss, tf.trainable_variables())
            train = optimiser.apply_gradients(grads_and_vars, global_step)
            with tf.control_dependencies([train]):
                global_step = tf.identity(global_step)
            return global_step
    return output


def initializer():
    utils.move_variable_initialization_to_cpu()
    return tf.global_variables_initializer()


def add_args(parser):
    parser.add_argument("--batch-size", default=1, type=int,
                        help="Number of inputs in a mini-batch")
    parser.add_argument("--filter-in", default=64, type=int,
                        help="Number of filters in the input")
    parser.add_argument("--filter-out", default=64, type=int,
                        help="Number of filters in the output")
    parser.add_argument("--kernel-size", default=3, type=int,
                        help="Kernel size")
    parser.add_argument("--stride", default=1, type=int,
                        help="Stride")
    parser.add_argument("--input-size", default=56, type=int,
                        help="Input size")
    parser.add_argument("--block-repeats", default=10, type=int,
                        help="Number of convolution blocks")
    parser.add_argument("--group-dim", default=1, type=int,
                        help="Group dimension (ie. number of filter in each group)")
    parser.add_argument("--train", action='store_true', dest='train',
                        help="Compute loss and optimization pass")
    parser.set_defaults(train=False, batches_per_step=1000, steps=5)
    return parser


def iteration_report(opts, time):
    return "{:5f} items/sec".format(opts.batch_size * opts.batches_per_step * opts.block_repeats / time)


if __name__ == '__main__':
    # Add benchmark module to path
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    sys.path.insert(1, os.path.join(cwd, '..', '..', '..', 'utils',
                                    'benchmarks', 'tensorflow'))
    import benchmark

    module = benchmark.Benchmark(
        graph_builder,
        inputs,
        initializer,
        add_args,
        iteration_report
    )

    options = benchmark.parse_opts(module, False)

    if options.shards > 0:
        raise NotImplementedError(
            "--shards option has not been implemented with this example")

    # Log Benchmark Message
    print("CNN layer {} Synthetic benchmark.\n"
          " Batch size {}.\n"
          " Batches per Step {}.\n"
          " Steps {}.\n"
          " Input size {}.\n"
          " Group dimension {}.\n"
          " Output filters {}.\n"
          " Block repeat {}.\n"
          .format(
              "Training" if options.train else "Inference",
              options.batch_size,
              options.batches_per_step if not options.cycle_report else "n/a",
              options.steps if not options.cycle_report else "n/a",
              options.input_size,
              options.group_dim,
              options.filter_out,
              options.block_repeats))

    benchmark.run(module, options)
