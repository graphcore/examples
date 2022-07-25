# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf
import tqdm
from tensorflow.python import ipu

from static_ops.static_ops import grouped_gather, grouped_scatter


@dataclasses.dataclass
class Flags:
    micro_batch_size: int = 8
    n_nodes: int = 24
    n_edges: int = 50
    n_latent: int = 128
    steps_per_epoch: int = 1024


def gather_func(x, indices, grouped=True):
    x = tf.cast(x, tf.float32)
    if grouped:
        gathered = grouped_gather(x, indices)
    else:
        outs = []
        for _x, _index in zip(x, indices):
            outs.append(tf.gather(_x, _index))

        gathered = tf.stack(outs)
    return gathered


def scatter_func(x, indices, grouped=True, n_nodes=24):
    if grouped:
        scattered = grouped_scatter(x, indices, table_size=n_nodes)
    else:
        outs = []
        for _x, _indices in zip(x, indices):
            outs.append(tf.math.unsorted_segment_sum(_x, _indices, num_segments=n_nodes))

        scattered = tf.stack(outs)
    return scattered


@pytest.mark.usefixtures("ipu_static_ops")
def test_grouped_scatter_gather():
    FLAGS = Flags()

    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    ipu.utils.configure_ipu_system(config)

    gather_grads = dict()
    scatter_grads = dict()

    N_TESTS = 10
    for _ in tqdm.trange(N_TESTS):
        inputs = tf.random.normal([FLAGS.micro_batch_size, FLAGS.n_nodes, FLAGS.n_latent])
        indices = tf.constant(np.random.randint(
            low=0, high=FLAGS.n_nodes, size=[FLAGS.micro_batch_size, FLAGS.n_edges]).astype(np.int32))

        for grouped in (True, False):
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                outputs = gather_func(inputs, indices, grouped=grouped)
                loss = tf.reduce_mean(tf.abs(outputs))
            gather_grads[grouped] = tape.gradient(loss, inputs)

            with tf.GradientTape() as tape:
                tape.watch(outputs)
                scattered = scatter_func(outputs, indices, grouped=grouped)
                loss = tf.reduce_mean(tf.abs(scattered))
            scatter_grads[grouped] = tape.gradient(loss, outputs)

        assert np.allclose(gather_grads[True], gather_grads[False]), "gather grads wrong"
        assert np.allclose(scatter_grads[True], scatter_grads[False]), "scatter grads wrong"
