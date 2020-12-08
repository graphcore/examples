#!/usr/bin/env python
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
Benchmark a single Dense layer.

The Items/sec reported at the end of the benchmark is based on wall time.

Run with -h or --help for options.
"""
import inspect
import os
import sys
import tensorflow as tf
from tensorflow.python.ipu import utils


def dense(opts, inputs):
    # Add ReLU activation function if appropriate option is set
    if opts.activation:
        return tf.layers.dense(units=opts.size, inputs=inputs, activation=tf.nn.relu)

    else:
        return tf.layers.dense(units=opts.size, inputs=inputs)


def inputs(opts, index):
    value = tf.cast(index, tf.float16)
    return {
        "inputs": tf.broadcast_to(value, [opts.batch_size, opts.size]),
    }


def graph_builder(opts, inputs):
    output = dense(opts, inputs["inputs"])

    if opts.train:
        # Loss is the mean across output matrix:
        loss = tf.reduce_mean(output)
        optimiser = tf.train.GradientDescentOptimizer(0.01)
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
    parser.add_argument("--batch-size", default=32, type=int,
                        help="Number of inputs in a mini-batch")
    parser.add_argument("--size", default=1024, type=int,
                        help="Dense layer size")
    parser.add_argument("--train", action='store_true', dest='train',
                        help="Compute loss and optimization pass")
    parser.add_argument("--include-activation", action='store_true', dest='activation',
                        help="Include ReLU activation (otherwise linear/no activation")

    parser.set_defaults(train=False, batches_per_step=5000, steps=5)
    return parser


def iteration_report(opts, time):
    return "{:5f} items/sec".format(opts.batch_size * opts.batches_per_step * opts.replicas / time)


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

    if options.shards > 1:
        raise NotImplementedError(
            "--shards option has not been implemented with this example")

    # Log Benchmark Message
    print(" Dense layer {} Synthetic benchmark.\n"
          " Batch size {}.\n"
          " Batches per Step {}.\n"
          " Dense size {}.\n"
          .format(
              "Training" if options.train else "Inference",
              options.batch_size,
              options.batches_per_step if not options.report else "n/a",
              options.size))

    benchmark.run(module, options)
