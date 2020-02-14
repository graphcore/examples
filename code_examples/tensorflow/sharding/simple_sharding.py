# Copyright 2019 Graphcore Ltd.
import argparse

import numpy as np
import tensorflow as tf

from tensorflow.python.ipu import autoshard, ipu_compiler, scopes, utils

parser = argparse.ArgumentParser()
parser.add_argument("--autoshard", action="store_true",
                    help="Enables automatic sharding")
parser.set_defaults(autoshard=False)
opts = parser.parse_args()

NUM_SHARDS = 2

# With sharding all placeholders MUST be explicitly placed on
# the CPU device:
with tf.device("cpu"):
    pa = tf.placeholder(np.float32, [2], name="a")
    pb = tf.placeholder(np.float32, [2], name="b")
    pc = tf.placeholder(np.float32, [2], name="c")


# Put part of the computation on shard 1 and part on shard 2.
# Sharding is automatically enabled on detection of nodes
# placed with 'scopes.ipu_shard(...)':
def manual_sharding(pa, pb, pc):
    with scopes.ipu_shard(0):
        o1 = pa + pb
    with scopes.ipu_shard(1):
        o2 = pa + pc
        out = o1 + o2
        return out


def auto_sharding(pa, pb, pc):
    # This context marks the section of the graph to autoshard.
    # In this case we want to autoshard across the whole graph
    # so this context isn't actually required.
    with autoshard.ipu_autoshard():
        o1 = pa + pb
        o2 = pa + pc
        out = o1 + o2
        return out


def my_graph(pa, pb, pc):
    if opts.autoshard:
        result = auto_sharding(pa, pb, pc)
        # The first argument to automatic_sharding is the number
        # of shards.  The second argument is the tensor closest to
        # the input data source in the graph.  In this case it
        # could be pa, pb or pc.  The third argument is the
        # tensor closest to the loss of the graph.  There is no
        # loss function, thus the output of the graph is the
        # closest.  By defining the extremities of the graph
        # the automatic sharding mechanism can calculate which
        # edges it can split across.
        autoshard.automatic_sharding(NUM_SHARDS, pa, result)
    else:
        result = manual_sharding(pa, pb, pc)
    return result

# Create the IPU section of the graph
with scopes.ipu_scope("/device:IPU:0"):
    out = ipu_compiler.compile(my_graph, [pa, pb, pc])

# Define the feed_dict input data
fd = {pa: [1., 1.], pb: [0., 1.], pc: [1., 5.]}
# Configure an IPU device that has NUM_SHARDS devices that we will
# shard across.
cfg = utils.create_ipu_config(profiling=True)
cfg = utils.auto_select_ipus(cfg, NUM_SHARDS)
utils.configure_ipu_system(cfg)

with tf.Session() as sess:
    result = sess.run(out, fd)
    print(result)
