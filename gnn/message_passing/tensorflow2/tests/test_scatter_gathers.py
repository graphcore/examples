# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python import ipu

from layers import _batched_segment_mean, _gather, _scatter


@pytest.mark.usefixtures("ipu_static_ops")
def test_scatter_gather():
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.device_connection.type = ipu.utils.DeviceConnectionType.ON_DEMAND
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

        out_1 = _gather(nodes, senders, gather_scatter_method='debug')
        out_2 = _gather(nodes, senders, gather_scatter_method='grouped')
        out_3 = _gather(nodes, one_hot_senders, gather_scatter_method='dense')

        assert np.allclose(out_1, out_2), "grouped gather is wrong"
        assert np.allclose(out_1, out_3), "dense gather is wrong"

        r1 = _scatter(out_1, receivers, gather_scatter_method='debug', num_segments=n_nodes)
        r2 = _scatter(out_1, receivers, gather_scatter_method='grouped', num_segments=n_nodes)
        r3 = _scatter(out_1, one_hot_receivers, gather_scatter_method='dense', num_segments=n_nodes)

        assert np.allclose(r1, r2), "grouped scatter is wrong"
        assert np.allclose(r1, r3), "dense scatter is wrong"

    print(f"Passed {n_tests}/{n_tests} tests: all implementations of scatter/gather are producing equivalent results.")


@pytest.mark.usefixtures("ipu_static_ops")
def test_graphwise_means():
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.device_connection.type = ipu.utils.DeviceConnectionType.ON_DEMAND
    ipu.utils.configure_ipu_system(config)

    n_nodes = 16
    n_hidden = 128
    n_graphs = 3
    bs = 32
    n_tests = 10

    for _ in range(n_tests):
        nodes = tf.random.uniform([bs, n_nodes, n_hidden])
        node_graph_idx = np.random.randint(low=0, high=n_graphs, size=(bs, n_nodes), dtype=np.int32)
        node_graph_idx.sort(axis=1)

        one_hot_node_graph_idx = tf.one_hot(node_graph_idx.astype(np.int32), depth=n_graphs)
        out_1 = _batched_segment_mean(nodes, node_graph_idx, n_graphs, gather_scatter_method='debug')
        out_2 = _batched_segment_mean(nodes, tf.constant(node_graph_idx), n_graphs, gather_scatter_method='grouped')
        out_3 = _batched_segment_mean(nodes, one_hot_node_graph_idx, n_graphs, gather_scatter_method='dense')
        assert np.allclose(out_1, out_2), "grouped graphwise mean is wrong"
        assert np.allclose(out_1, out_3), "dense graphwise mean is wrong"

    print(f"Passed {n_tests}/{n_tests} tests: differing implementations of graphwise means are "
          f"producing equivalent results.")
