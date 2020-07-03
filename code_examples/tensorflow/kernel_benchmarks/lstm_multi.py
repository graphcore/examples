#!/usr/bin/env python
# Copyright 2019 Graphcore Ltd.
"""
Benchmark a multi-layer LSTM model with a dense final layer.

The Items/sec reported at the end of the benchmark is based on wall time.

Run with -h or --help for options.
"""
import inspect
import os
import sys
import tensorflow as tf
from tensorflow.python.ipu import utils
from tensorflow.python.ipu import rnn_ops


def lstm_block(input_tensor, num_units, opts, name=""):
    # PopnnLSTM uses a direct poplibs implementation
    lstm_cell = rnn_ops.PopnnLSTM(num_units=num_units, dtype=input_tensor.dtype, name=name)
    # The input is [timesteps, batch_size, input_size]
    return lstm_cell(input_tensor, training=opts.train)


def lstm_model(opts, inputs):
    # Add LSTM layers
    x = inputs
    for _ in range(opts.num_layers):
        x, final_state = lstm_block(x, opts.hidden_size, opts)
    # final_state.h contains the last output of LSTM
    x = final_state.h
    # Add a final dense layer
    x = tf.layers.dense(x, 1)
    return x


def inputs(opts, index):
    value = tf.cast(index, tf.float16)
    return {
        "inputs": tf.broadcast_to(value, [opts.timesteps, opts.batch_size, opts.input_size]),
        "labels": tf.broadcast_to(value, [opts.batch_size, 1]),
    }


def graph_builder(opts, inputs):
    output = lstm_model(opts, inputs["inputs"])

    if opts.train:
        # Loss
        loss = tf.losses.mean_squared_error(inputs["labels"], output)
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
    parser.add_argument("--batch-size", default=1, type=int,
                        help="Number of inputs in a mini-batch")
    parser.add_argument("--timesteps", default=200, type=int,
                        help="Number of recurrent steps")
    parser.add_argument("--input-size", default=16, type=int,
                        help="Input size")
    parser.add_argument("--hidden-size", default=256, type=int,
                        help="LSTM hidden size")
    parser.add_argument("--num-layers", default=2, type=int,
                        help="LSTM number of layers")
    parser.add_argument("--train", action='store_true', dest='train',
                        help="Compute loss and optimization pass")
    parser.add_argument("--no-train", action='store_false', dest='train',
                        help="No loss or optimisation pass will be computed (default)")
    parser.set_defaults(train=False, batches_per_step=1000, steps=5)
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
    print("Multi-layer LSTM with a dense final layer, {} Benchmark.\n"
          " Batch size {}.\n"
          " Batches per Step {}.\n"
          " Steps {}.\n"
          " Hidden size {}.\n"
          " Number of layers {}.\n"
          " Timesteps {}.\n"
          .format(
              "Training" if options.train else "Inference",
              options.batch_size,
              options.batches_per_step if not options.report else "n/a",
              options.steps if not options.report else "n/a",
              options.hidden_size,
              options.num_layers,
              options.timesteps))

    benchmark.run(module, options)
