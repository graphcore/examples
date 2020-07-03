# Copyright 2019 Graphcore Ltd.
import argparse
import numpy as np
import tensorflow as tf

from tensorflow.python import ipu

# Handle CMD arguments
parser = argparse.ArgumentParser()
parser.add_argument('--replication-factor', type=int, default=2,
                    help="Number of IPUs to replicate across")
parser.add_argument('--num-data-points', type=int, default=50,
                    help="Number of data points in the toy linear regression datset")
parser.add_argument('--num-features', type=int, default=100,
                    help="Number of features in the data")
parser.add_argument('--num-iters', type=int, default=250,
                    help="Number of iterations")
opts = parser.parse_args()

# Make a simple linear regression tf Dataset, of N noisy x = y lines, squashed into range [0, 1]
fx = np.tile(np.linspace(0, 1, opts.num_features), [opts.num_data_points, 1])
x = (fx + np.random.uniform(-1, 1, fx.shape)).astype(np.float32)[:, None]
y = (fx + np.random.uniform(-1, 1, fx.shape)).astype(np.float32)[:, None]
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.map(lambda x, y: (tf.nn.sigmoid(x),
                                    tf.nn.sigmoid(y)))
dataset = dataset.repeat()
# Make the IPU infeed and outfeed
# To use replication, we make as many feeds as there are replicated IPUs by passing in replication_factor
infeed = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, replication_factor=opts.replication_factor, feed_name='in')
outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue(replication_factor=opts.replication_factor, feed_name='out')


# Make a basic linear model
def model(X, Y):
    Yp = tf.layers.dense(X, opts.num_features)
    loss = tf.losses.mean_squared_error(Y, Yp)
    optimizer = tf.train.GradientDescentOptimizer(1e-3)
    # To use replication, we wrap our optimizer with the IPU custom CrossReplicaOptimizer,
    # ...which averages the gradients determined by all IPUs together
    training_op = ipu.cross_replica_optimizer.CrossReplicaOptimizer(optimizer).minimize(loss)
    # We can also use the CrossReplicaGradientAccumulationOptimizer instead, which accumulates gradients
    # ...every N mini_batches before updating parameters, to effectively increase the batch size
    # ...For replication, this reduces the number of inter-IPU synchs by the factor N.
    # training_op = ipu.gradient_accumulation_optimizer.CrossReplicaGradientAccumulationOptimizer(optimizer, num_mini_batches=8).minimize(loss)
    # Enqueue the loss to be dequeued later off the IPU
    return outfeed.enqueue(loss), training_op


# Repeat the training 250 times on the IPU (i.e. with no switch back to the host)
def training_loop():
    return ipu.loops.repeat(opts.num_iters, model, infeed_queue=infeed)


# Compile the graph with the IPU custom xla compiler
with ipu.scopes.ipu_scope("/device:IPU:0"):
    compiled = ipu.ipu_compiler.compile(training_loop)

# Ops to read the outfeed and initialize all variables
dequeue_outfeed_op = outfeed.dequeue()
init_op = tf.global_variables_initializer()

# Configure the IPU
# 'max_cross_replica_sum_buffer_size' determines the amount in memory of gradients to
# ...accumulate before updating parameters, when using CrossReplicaGradientAccumulationOptimizer
# ...Increasing this will reduce IPU memory but increase performance
cfg = ipu.utils.create_ipu_config(profiling=False, max_cross_replica_sum_buffer_size=10000000)  # 10mb
# Auto select as many IPUs as we want to replicate across
# ...(must be a power of 2 - IPU driver MultiIPUs come only in powers of 2)
cfg = ipu.utils.auto_select_ipus(cfg, opts.replication_factor)
ipu.utils.configure_ipu_system(cfg)

# Run the model
with tf.Session() as sess:
    # Initialize
    sess.run(init_op)
    sess.run(infeed.initializer)
    # Run
    sess.run(compiled)
    # Read the outfeed for the training losses
    losses = sess.run(dequeue_outfeed_op)
    # Average the losses over the replicated IPUs
    print(f"Losses:\n{np.mean(losses, axis=1)}")
