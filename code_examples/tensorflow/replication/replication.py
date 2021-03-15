# Copyright 2020 Graphcore Ltd.
import argparse
from functools import partial
import os
import time

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Dropout,
                                     Flatten, MaxPooling2D)

from tensorflow.python import ipu


NUM_CLASSES = 10


"""
The following diagram shows a high-level representation of the underlying
mechanics of replication in Poplar and TensorFlow, when replicating N times:

                              Data pipeline
                                    |
                                    |
                                    v                                  <------+
      +------------------------------------------------------------+          |
      |               |                            |               |          |
      |               |                            |               |          |
      v               v                            v               v          |
    Model           Model                        Model           Model        |
      |               |                            |               |          |
      |               |                            |               |          |
      v               v                            v               v          |
  Gradients0      Gradients1                  GradientsN-1     GradientsN     |
    | | |           | | |                        | | |           | | |        |
    | | |           | | |                        | | |           | | |  <--+  |
    v v v           v v v                        v v v           v v v     |  |
  +-------+       +-------+                    +-------+       +-------+   |  |
  |       |       |       |       Cross        |       |       |       |   |  |
  |       |       |       |      Replica       |       |       |       |   |  |
  |       |       |       |      Buffers       |       |       |       |   |  |
  +-------+       +-------+                    +-------+       +-------+   |  |
      |               |                            |               |       |  |
      |               |                            |               |       |  |
      v               v                            v               v       |  |
  ============================= All reduce =============================   |  |
      |               |                            |               |       |  |
      +---------------+----------------------------+---------------+-------+  |
      |               |                            |               |          |
      |               |                            |               |          |
weight update   weight update                weight update   weight update    |
      |               |                            |               |          |
      |               |                            |               |          |
      +---------------+----------------------------+---------------+----------+

A single TensorFlow data pipeline is wrapped by an IPU infeed queue, which
serves each of the replicas concurrently as they request pipeline elements.
Replicas calculate the loss for their data elements, then calculate gradients
with respect to that loss. The replicas then synchronise and do an all-reduce on
the gradients, averaging them. By default, each gradient is reduced separately
in sequence.

The all-reduce itself is implemented as a ring all-reduce - see
https://en.wikipedia.org/wiki/All-to-all_(parallel_pattern)#Ring. If the
all-reduce is on a gradient containing G elements over N replicas, this involves
O(N) global exchanges, where each global exchange involves each replica
exchanging O(G/N) elements with its neighbour.

You can control the amount of gradients all-reduced at a time by changing the
max_cross_replica_sum_buffer_size (specified in bytes). For example, if the
buffer size is 10MB, then 10MB of gradients (at most) will be exchanged in a
single all-reduce. Note that changing this value and the number of replicas
represents an always-live vs not always-live trade off:
  - Increasing the max_cross_replica_sum_buffer_size will lead to larger
    temporary buffers in the all-reduce exchange, but fewer all-reduces overall
    and therefore less control code.
  - Increasing the number of replicas will lead to smaller temporary buffers in
    the all-reduce exchange, but more exchanges in each all-reduce and therefore
    more control code.

Some additional notes:
  - If the temporary buffers are too large, then their exchanges may start
    generating more control code than they save.
  - If your model contains a lot of trainable variables, it's strongly advised
    to consider adjusting the max_cross_replica_sum_buffer_size.
  - If a model doesn't fit with a small number of replicas due to memory spikes,
    you may find a larger number of replicas does fit.
"""


def get_dataset(opts):
    # Since each replica consumes data from the dataset individually,
    # it is important for the data pipeline to be able to push data at the
    # speeds that the replicas require, otherwise the model will be data-bound
    train_data, _ = cifar10.load_data()
    X = train_data[0].astype("float32") / 255.0
    y = train_data[1].astype("int32")

    def generator():
        return zip(X, y)

    types = (X.dtype, y.dtype)
    shapes = (X.shape[1:], y.shape[1:])

    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(len(X))
    dataset = dataset.batch(opts.batch_size, drop_remainder=True)
    return dataset


def training_step(opts, outfeed, X, y):
    # Build a simple convolutional model with the Keras API
    model = Sequential()
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))

    # In TensorFlow 1, the TensorFlow Optimizer API must be used to train Keras
    # models
    logits = model(X, training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

    # To use replication, we wrap our optimizer with the
    # CrossReplicaOptimizer, which averages the gradients of all IPUs together
    # If we also want to accumulate gradients, we can use the
    # CrossReplicaGradientAccumulationOptimizerV2
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=opts.learning_rate
    )
    if opts.batches_to_accumulate > 1:
        optimizer = ipu.optimizers. \
            CrossReplicaGradientAccumulationOptimizerV2(
                optimizer,
                num_mini_batches=opts.batches_to_accumulate)
    else:
        optimizer = ipu.optimizers.CrossReplicaOptimizer(optimizer)

    train_op = optimizer.minimize(loss)

    # Returned ops are executed as control dependencies
    return outfeed.enqueue(loss), train_op


def build_IPU_graph(opts, dataset):
    # Set up infeeds and outfeeds on the host to streamcopy batch-wise data
    # to/from the IPU during each IPU step
    # They must be aware of the number of replicas
    infeed = ipu.ipu_infeed_queue.IPUInfeedQueue(
        dataset,
        replication_factor=opts.replicas,
        feed_name='infeed')
    outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
        replication_factor=opts.replicas,
        feed_name='outfeed')

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        # Repeat the model n times on the IPU before returning to the host
        def training_loop():
            # The infeed unpacks its values into the training_step's remaining
            # arguments as X and y. Each replica consumes an element from the
            # infeed asynchronously every iteration.
            return ipu.loops.repeat(opts.iterations_per_step,
                                    partial(training_step, opts, outfeed),
                                    infeed_queue=infeed)
        compile_and_run = ipu.ipu_compiler.compile(training_loop)

    # Create a graph op to dequeue the contents of the outfeed
    dequeue_outfeed = outfeed.dequeue()

    # Create a graph op to initialise all variables
    init = tf.global_variables_initializer()
    # Infeed must be initialised just like any other Dataset
    init = tf.group([init, infeed.initializer])
    # Make the CPU initialize the graph, then streamcopy the weights to the IPU
    ipu.utils.move_variable_initialization_to_cpu()

    # Configure the IPU
    config = ipu.utils.create_ipu_config()
    # We should request as many IPUs as all of the replicas need. Since the
    # model requires 1 IPU, and we replicate it N times, we need N IPUs.
    config = ipu.utils.auto_select_ipus(config, opts.replicas)
    # Set the max_cross_replica_sum_buffer_size
    config = ipu.utils.set_optimization_options(
        config,
        max_cross_replica_sum_buffer_size=opts.max_cross_replica_sum_buffer_size
    )
    ipu.utils.configure_ipu_system(config)

    return init, compile_and_run, dequeue_outfeed


if __name__ == '__main__':
    # Get command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('--replicas', type=int, default=2,
                        help="Number of times to replicate the graph")
    parser.add_argument('--max-cross-replica-sum-buffer-size', type=int,
                        default=10*1024**2,
                        help="Set the amount of gradients to all-reduce at a"
                             " time.")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--iterations-per-step', type=int, default=250,
                        help="Number of iterations to perform on the IPU"
                             " before returning control to the host.")
    parser.add_argument('--batches-to-accumulate', type=int, default=1,
                        help="Number of batches to accumulate (per replica)"
                             " before performing a weight update. 1 = no"
                             " accumulation. Note: this is gradient"
                             " accumulation, a concept separate from"
                             " replication, but here we show they can easily"
                             " be used together. The number of batches to"
                             " accumulate must evenly divide, and be smaller"
                             " than, the number of iterations per step.")
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--steps', type=int, default=500,
                        help="Number of steps to train.")
    parser.add_argument('--use-synthetic-data', action='store_true',
                        help="Whether to use synthetic data - data generated"
                             " on the IPU as needed with no host I/O.")
    opts = parser.parse_args()
    if opts.batches_to_accumulate > opts.iterations_per_step:
        raise ValueError(
            "Cannot accumulate more batches than there are iterations.")
    if opts.iterations_per_step % opts.batches_to_accumulate != 0:
        raise ValueError(
            "Undefined behaviour when number of batches to accumulate doesn't"
            " evenly divide the number of iterations - an accumulation could be"
            " interrupted by program termination.")

    # Create the data pipeline
    data = get_dataset(opts)

    if opts.use_synthetic_data:
        # Ignore data in the data pipeline, do no host I/O and generate data
        # on the device as needed
        os.environ["TF_POPLAR_FLAGS"] = (os.environ.get("TF_POPLAR_FLAGS", '') +
                                         " --use_synthetic_data ")

    # Create the graph
    init, compile_and_run, dequeue_outfeed = build_IPU_graph(opts, data)

    # Since each replica consumes a batch asynchronously every iteration,
    # and since a weight update happens at minimum after a single iteration,
    # replication scales the model's effective batch size
    effective_batch_size = opts.batch_size * opts.replicas
    # Gradient accumulation also naturally increases the effective batch size
    effective_batch_size *= opts.batches_to_accumulate
    # Learning rate should be modified in line with the effective batch size
    opts.learning_rate *= effective_batch_size

    with tf.Session() as sess:
        # Initialize the graph
        sess.run(init)

        for i in range(opts.steps):
            # Run and retrieve the losses from the outfeed
            # The graph is compiled the first time it is run
            sess.run(compile_and_run)

            # The loss for each replica, for each iteration in the step is
            # returned in an (iterations_per_step, replicas) array
            losses = sess.run(dequeue_outfeed)
            print(f"Step: {i+1} | Average loss: {np.mean(losses):.3f}")
