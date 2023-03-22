# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python import ipu

from model.gnn.aggregators import _gather, _scatter_max, _scatter_mean, _scatter_softmax, _scatter_sqrtN, _scatter_sum


@pytest.mark.usefixtures("ipu_static_ops")
def test_scatter_gather():
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
    ipu.utils.configure_ipu_system(config)

    n_tests = 10
    n_nodes = 6
    n_edges = 8
    n_hidden = 16
    bs = 128

    for _ in range(n_tests):
        nodes = tf.random.uniform([bs, n_nodes, n_hidden])
        indices = tf.random.uniform(shape=[bs, n_edges, 2], maxval=n_nodes, dtype=tf.int32)
        senders, receivers = indices[..., 0], indices[..., 1]

        one_hot_senders = tf.one_hot(senders, depth=n_nodes)
        one_hot_receivers = tf.one_hot(receivers, depth=n_nodes)

        out_1 = _gather(nodes, senders, gather_scatter_method="debug")
        out_2 = _gather(nodes, senders, gather_scatter_method="grouped")
        out_3 = _gather(nodes, one_hot_senders, gather_scatter_method="dense")

        assert np.allclose(out_1, out_2), "grouped gather is wrong"
        assert np.allclose(out_1, out_3), "dense gather is wrong"

        r1 = _scatter_sum(out_1, receivers, gather_scatter_method="debug", num_segments=n_nodes)
        r2 = _scatter_sum(out_1, receivers, gather_scatter_method="grouped", num_segments=n_nodes)
        r3 = _scatter_sum(out_1, one_hot_receivers, gather_scatter_method="dense", num_segments=n_nodes)

        assert np.allclose(r1, r2), "grouped scatter is wrong"
        assert np.allclose(r1, r3), "dense scatter is wrong"

    print(f"Passed {n_tests}/{n_tests} tests: all implementations of scatter/gather are producing equivalent results.")


@pytest.mark.usefixtures("ipu_static_ops")
def test_scatter_max():
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
    ipu.utils.configure_ipu_system(config)

    n_tests = 10
    n_nodes = 8
    n_edges = 4
    n_hidden = 4
    bs = 8

    for i in range(n_tests):
        edges = 2 * tf.random.uniform([bs, n_edges, n_hidden], dtype=tf.float16) - 1
        indices = tf.random.uniform(shape=[bs, n_edges], maxval=n_nodes, dtype=tf.int32)

        r1 = _scatter_max(edges, indices, n_nodes, "grouped")
        r2 = _scatter_max(edges, indices, n_nodes, "debug")

        assert np.allclose(r1, r2), "grouped scatter is wrong"
    print(f"Passed {n_tests}/{n_tests} tests: all implementations of scatter_max are producing equivalent results.")


@pytest.mark.usefixtures("ipu_static_ops")
def test_scatter_softmax():
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
    ipu.utils.configure_ipu_system(config)

    n_tests = 10
    n_nodes = 8
    n_edges = 4
    n_hidden = 4
    bs = 4

    for i in range(n_tests):
        edges = 2 * tf.random.uniform([bs, n_edges, n_hidden], dtype=tf.float16) - 1
        indices = tf.random.uniform(shape=[bs, n_edges], maxval=n_nodes, dtype=tf.int32)

        r1 = _scatter_softmax(edges, indices, n_nodes, gather_scatter_method="grouped")
        r2 = _scatter_softmax(edges, indices, n_nodes, gather_scatter_method="debug")

        assert np.allclose(r1, r2, atol=5e-4), "grouped scatter is wrong"
    print(f"Passed {n_tests}/{n_tests} tests: all implementations of scatter_max are producing equivalent results.")


@pytest.mark.usefixtures("ipu_static_ops")
def test_scatter_mean():
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
    ipu.utils.configure_ipu_system(config)

    n_tests = 10
    n_nodes = 8
    n_edges = 4
    n_hidden = 4
    bs = 4

    for i in range(n_tests):
        edges = 2 * tf.random.uniform([bs, n_edges, n_hidden], dtype=tf.float16) - 1
        indices = tf.random.uniform(shape=[bs, n_edges], maxval=n_nodes, dtype=tf.int32)

        r1 = _scatter_mean(edges, indices, n_nodes, gather_scatter_method="grouped")
        r2 = _scatter_mean(edges, indices, n_nodes, gather_scatter_method="debug")

        assert np.allclose(r1, r2), "grouped scatter is wrong"
    print(f"Passed {n_tests}/{n_tests} tests: all implementations of scatter_max are producing equivalent results.")


@pytest.mark.usefixtures("ipu_static_ops")
def test_scatter_sqrtN():
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
    ipu.utils.configure_ipu_system(config)

    n_tests = 10
    n_nodes = 8
    n_edges = 4
    n_hidden = 4
    bs = 4

    for i in range(n_tests):
        edges = 2 * tf.random.uniform([bs, n_edges, n_hidden], dtype=tf.float16) - 1
        indices = tf.random.uniform(shape=[bs, n_edges], maxval=n_nodes, dtype=tf.int32)

        r1 = _scatter_sqrtN(edges, indices, n_nodes, gather_scatter_method="grouped")
        r2 = _scatter_sqrtN(edges, indices, n_nodes, gather_scatter_method="debug")

        assert np.allclose(r1, r2), "grouped scatter is wrong"
    print(f"Passed {n_tests}/{n_tests} tests: all implementations of scatter_max are producing equivalent results.")
