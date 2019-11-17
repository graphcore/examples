# Copyright 2019 Graphcore Ltd.
import inspect
import os
import sys
import tensorflow as tf

from tensorflow.python.ipu import utils

# Add path with models and benchmarks folder to pythonpath
cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
sys.path.insert(1, os.path.join(cwd, '..'))
sys.path.insert(1, os.path.join(cwd, '..', '..', '..', '..', 'utils',
                                'benchmarks', 'tensorflow'))

from models import TensorflowResNet


class OptimizedResNet(TensorflowResNet):
    def __init__(self, opts):
        if not opts.train and opts.norm_type == 'BATCH':
            # For inference, assume normalization on population parameters
            # reduced to a single linear Ax+B transformation. Also assume that the
            # optimization has been applied that folds this transformation into
            # the previous conv + bias layer so there is no normalization op needed.
            opts.norm_type = 'NONE'
        super(OptimizedResNet, self).__init__(opts)


def graph_builder(opts, inputs):
    # Inference
    logits = OptimizedResNet(opts)(inputs["inputs"])

    # Loss
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=inputs['labels']))

    # Training
    if opts.train:
        global_step = tf.train.get_or_create_global_step()
        update_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy, global_step=global_step)
        with tf.control_dependencies([update_step]):
            global_step = tf.identity(global_step)
        return global_step
    else:
        return cross_entropy


def initializer():
    utils.move_variable_initialization_to_cpu()
    return tf.global_variables_initializer()


def inputs(opts, index):
    return {
        "inputs": tf.broadcast_to(tf.cast(index, tf.float16), [opts.batch_size, 224, 224, 4]),
        "labels": tf.broadcast_to(tf.cast(index, tf.int32), [opts.batch_size]),
    }


def add_args(parser):
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Set batch size.')
    parser.add_argument('--train', action="store_true",
                        help='Run training benchmark.')
    parser.add_argument('--size', type=int, choices=[8, 14, 20, 32, 44, 56, 110, 18, 34, 50, 101, 152], default=18,
                        help='Size of ResNet graph.')
    parser.add_argument('--norm-type', choices=["BATCH", "GROUP", "NONE"], default="BATCH",
                        help="Choose which normalization to use after each convolution")
    parser.add_argument('--norm-groups', type=int, default=32,
                        help="Sets the number of groups when using the 'GROUP' norm-type")
    parser.add_argument('--shortcut-type', choices=['A', 'B', 'C'],
                        help="ResNet shortcut type. Defaults to definition specified.")
    parser.set_defaults(batches_per_step=1000, steps=5, convolution_options=None)
    return parser


def iteration_report(opts, time):
    return "{:5f} images/sec.".format(opts.batch_size * opts.batches_per_step / time)


if __name__ == '__main__':
    import benchmark

    module = benchmark.Benchmark(
        graph_builder,
        inputs,
        initializer,
        add_args,
        iteration_report
    )

    opts = benchmark.parse_opts(module, False)

    if opts.shards > 0:
        raise NotImplementedError("--shards option has not been implemented with this example")

    # Temporary setting for ResNet18 before this option is automated.
    if opts.size == 18:
        opts.convolution_options = '{"availableMemoryProportion": "0.4"}'

    # Log Benchmark Message
    print("TensorFlow ResNet{} {} Synthetic benchmark.\n"
          " Batch size {}.\n"
          " Batches per Step {}.\n"
          " Steps {}."
          .format(
              opts.size,
              "Training" if opts.train else "Inference",
              opts.batch_size,
              opts.batches_per_step if not opts.cycle_report else "n/a",
              opts.steps if not opts.cycle_report else "n/a"))

    benchmark.run(module, opts)
