# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf
import tqdm
from tensorflow.python import ipu

from model.gnn.aggregators import _gather, _scatter_max, _scatter_mean, _scatter_softmax, _scatter_sqrtN, _scatter_sum


@dataclasses.dataclass
class Flags:
    micro_batch_size: int = 8
    n_nodes: int = 24
    n_edges: int = 50
    n_latent: int = 128
    n_tests: int = 5


@pytest.mark.usefixtures("ipu_static_ops")
def test_grouped_scatter_gather():
    FLAGS = Flags()

    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
    ipu.utils.configure_ipu_system(config)

    gather_grads = dict()
    scatter_grads = dict()

    N_TESTS = FLAGS.n_tests
    for _ in tqdm.trange(N_TESTS):
        inputs = tf.random.normal([FLAGS.micro_batch_size, FLAGS.n_nodes, FLAGS.n_latent])
        indices = tf.constant(
            np.random.randint(low=0, high=FLAGS.n_nodes, size=[FLAGS.micro_batch_size, FLAGS.n_edges]).astype(np.int32)
        )

        for grouped in (True, False):
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                outputs = _gather(inputs, indices, gather_scatter_method="grouped" if grouped else "debug")
                loss = tf.reduce_mean(tf.abs(outputs))
            gather_grads[grouped] = tape.gradient(loss, inputs)

            with tf.GradientTape() as tape:
                tape.watch(outputs)
                scattered = _scatter_sum(
                    outputs, indices, FLAGS.n_nodes, gather_scatter_method="grouped" if grouped else "debug"
                )
                loss = tf.reduce_mean(tf.abs(scattered))
            scatter_grads[grouped] = tape.gradient(loss, outputs)

        assert np.allclose(gather_grads[True], gather_grads[False]), "gather grads wrong"
        assert np.allclose(scatter_grads[True], scatter_grads[False]), "scatter grads wrong"


@pytest.mark.usefixtures("ipu_static_ops")
def test_grouped_scatter_max():
    FLAGS = Flags()

    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
    ipu.utils.configure_ipu_system(config)

    scatter_grads = dict()

    N_TESTS = FLAGS.n_tests
    for i in tqdm.trange(N_TESTS):
        # half random half identical to catch corner cases in bwd pass
        inputs1 = 2 * tf.random.normal([FLAGS.micro_batch_size, FLAGS.n_edges, FLAGS.n_latent // 2]) - 1
        inputs2 = tf.ones([FLAGS.micro_batch_size, FLAGS.n_edges, FLAGS.n_latent // 2]) - 0.5
        inputs = tf.concat([inputs1, inputs2], axis=-1)
        inputs = tf.cast(inputs, tf.float16)
        indices = tf.constant(
            np.random.randint(low=0, high=FLAGS.n_nodes, size=[FLAGS.micro_batch_size, FLAGS.n_edges]).astype(np.int32)
        )

        for grouped in (True, False):
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                scattered = _scatter_max(
                    inputs,
                    indices,
                    FLAGS.n_nodes,
                    gather_scatter_method="grouped" if grouped else "debug",
                    backwards_mode="mean",
                )
                loss = tf.reduce_mean(tf.abs(scattered))
            scatter_grads[grouped] = tape.gradient(loss, inputs)

        # print(f"test {i}", scatter_grads[True], scatter_grads[False], scatter_grads[True] - scatter_grads[False])
        assert np.allclose(scatter_grads[True], scatter_grads[False]), "scatter grads wrong"


@pytest.mark.usefixtures("ipu_static_ops")
def test_grouped_scatter_softmax():
    FLAGS = Flags()

    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
    ipu.utils.configure_ipu_system(config)

    scatter_grads = dict()

    N_TESTS = FLAGS.n_tests
    for i in tqdm.trange(N_TESTS):
        # half random half identical to catch corner cases in bwd pass
        inputs1 = 2 * tf.random.normal([FLAGS.micro_batch_size, FLAGS.n_edges, FLAGS.n_latent // 2]) - 1
        inputs2 = tf.ones([FLAGS.micro_batch_size, FLAGS.n_edges, FLAGS.n_latent // 2]) - 0.5
        inputs = tf.concat([inputs1, inputs2], axis=-1)
        inputs = tf.cast(inputs, tf.float16)
        indices = tf.constant(
            np.random.randint(low=0, high=FLAGS.n_nodes, size=[FLAGS.micro_batch_size, FLAGS.n_edges]).astype(np.int32)
        )

        for grouped in (True, False):
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                scattered = _scatter_softmax(
                    inputs, indices, FLAGS.n_nodes, stable=True, gather_scatter_method="grouped" if grouped else "debug"
                )
                loss = tf.reduce_mean(tf.abs(scattered))
            scatter_grads[grouped] = tape.gradient(loss, inputs)

        # print(f"test {i}", scatter_grads[True], scatter_grads[False], scatter_grads[True] - scatter_grads[False])
        assert np.allclose(scatter_grads[True], scatter_grads[False], atol=1e-4), "scatter grads wrong"


@pytest.mark.usefixtures("ipu_static_ops")
def test_grouped_scatter_mean():
    FLAGS = Flags()

    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
    ipu.utils.configure_ipu_system(config)

    scatter_grads = dict()

    N_TESTS = FLAGS.n_tests
    for i in tqdm.trange(N_TESTS):
        # half random half identical to catch corner cases in bwd pass
        inputs1 = 2 * tf.random.normal([FLAGS.micro_batch_size, FLAGS.n_edges, FLAGS.n_latent // 2]) - 1
        inputs2 = tf.ones([FLAGS.micro_batch_size, FLAGS.n_edges, FLAGS.n_latent // 2]) - 0.5
        inputs = tf.concat([inputs1, inputs2], axis=-1)
        inputs = tf.cast(inputs, tf.float16)
        indices = tf.constant(
            np.random.randint(low=0, high=FLAGS.n_nodes, size=[FLAGS.micro_batch_size, FLAGS.n_edges]).astype(np.int32)
        )

        for grouped in (True, False):
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                scattered = _scatter_mean(
                    inputs, indices, FLAGS.n_nodes, gather_scatter_method="grouped" if grouped else "debug"
                )
                loss = tf.reduce_mean(tf.abs(scattered))
            scatter_grads[grouped] = tape.gradient(loss, inputs)

        # print(f"test {i}", scatter_grads[True], scatter_grads[False], scatter_grads[True] - scatter_grads[False])
        assert np.allclose(scatter_grads[True], scatter_grads[False], atol=1e-4), "scatter grads wrong"


@pytest.mark.usefixtures("ipu_static_ops")
def test_grouped_scatter_sqrtN():
    FLAGS = Flags()

    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
    ipu.utils.configure_ipu_system(config)

    scatter_grads = dict()

    N_TESTS = FLAGS.n_tests
    for i in tqdm.trange(N_TESTS):
        # half random half identical to catch corner cases in bwd pass
        inputs1 = 2 * tf.random.normal([FLAGS.micro_batch_size, FLAGS.n_edges, FLAGS.n_latent // 2]) - 1
        inputs2 = tf.ones([FLAGS.micro_batch_size, FLAGS.n_edges, FLAGS.n_latent // 2]) - 0.5
        inputs = tf.concat([inputs1, inputs2], axis=-1)
        inputs = tf.cast(inputs, tf.float16)
        indices = tf.constant(
            np.random.randint(low=0, high=FLAGS.n_nodes, size=[FLAGS.micro_batch_size, FLAGS.n_edges]).astype(np.int32)
        )

        for grouped in (True, False):
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                scattered = _scatter_sqrtN(
                    inputs, indices, FLAGS.n_nodes, gather_scatter_method="grouped" if grouped else "debug"
                )
                loss = tf.reduce_mean(tf.abs(scattered))
            scatter_grads[grouped] = tape.gradient(loss, inputs)

        # print(f"test {i}", scatter_grads[True], scatter_grads[False], scatter_grads[True] - scatter_grads[False])
        assert np.allclose(scatter_grads[True], scatter_grads[False], atol=1e-4), "scatter grads wrong"
