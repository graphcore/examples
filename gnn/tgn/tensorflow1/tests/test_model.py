# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Any, Dict, Tuple

import numpy as np
import pytest
import tensorflow.compat.v1 as tf
import torch as T
import torch_geometric

import model


def test_assert_shape() -> None:
    x = tf.constant(2, shape=(10, 20, 30))
    model.assert_shape(x, (10, 20, 30))
    model.assert_shape(x, (10, None, 30))
    with pytest.raises(AssertionError):
        model.assert_shape(x, (10, 21, 30))
    with pytest.raises(AssertionError):
        model.assert_shape(x, (None, 10, 20, 30))


def test_scoped_fn() -> None:
    @model.scoped_fn
    def fancy_layer() -> tf.Variable:
        return tf.get_variable("weight", (10, ), tf.float32)

    with tf.Graph().as_default():
        weight = fancy_layer()
        assert weight.name == "fancy_layer/weight:0"


def test_index_softmax() -> None:
    values = tf.constant([0, 0, np.log(2), 2], tf.float32)
    indices = tf.constant([0, 0, 0, 1])
    expected = np.array([0.25, 0.25, 0.5, 1.0])
    with tf.Session() as session:
        np.testing.assert_allclose(
            session.run(model.index_softmax(values, indices, 2)), expected)
        np.testing.assert_allclose(
            session.run(model.index_softmax(values, indices, 10)), expected)
        np.testing.assert_allclose(
            session.run(
                model.index_softmax(tf.tile(values[:, np.newaxis], (1, 3)),
                                    indices, 2)),
            np.tile(expected[:, np.newaxis], (1, 3)),
        )


def test_linear() -> None:
    with tf.Graph().as_default(), tf.Session() as session, tf.variable_scope(
            "test", reuse=tf.AUTO_REUSE):
        output = model.linear(tf.constant([[1, 2, 3]], dtype=tf.float32),
                              2,
                              use_bias=True)
        session.run(tf.global_variables_initializer())
        session.run(
            tf.assign(tf.get_variable("linear/weight"),
                      [[0, 0], [1, 0], [0, 10]]))
        np.testing.assert_allclose(session.run(output), [[2, 30]])


def test_cos_fp16() -> None:
    xs = tf.constant([0, 0.4, np.pi, 0.4 + 2 * 1e5 * np.pi])
    expected = tf.cos(xs)
    actual = model.cos_fp16(xs)
    with tf.Session() as session:
        np.testing.assert_allclose(session.run(expected),
                                   session.run(actual),
                                   atol=1e-2)


def _set_variables(session: tf.Session, values: Dict[str, Any]) -> None:
    variables = {
        v.name: v
        for v in session.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    }
    for name, value in values.items():
        session.run(tf.assign(variables[name + ":0"], np.array(value)))


def _assert_allclose(actual: np.ndarray,
                     expected: np.ndarray,
                     padded: bool = False) -> None:
    if padded:
        assert len(actual.shape) == len(expected.shape)
        assert all(a >= b for a, b in zip(actual.shape, expected.shape))
        actual_unpadded = actual[tuple(slice(0, n) for n in expected.shape)]
    else:
        assert actual.shape == expected.shape
        actual_unpadded = actual
    np.testing.assert_allclose(actual_unpadded, expected, atol=1e-3)


def test_gru_cell() -> None:
    # Generate a reference output using PyTorch
    T.manual_seed(8691)
    module = T.nn.GRUCell(5, 8)
    input, hidden = T.randn(2, 5), T.randn(2, 8)
    ref_out = module(input, hidden)

    tf_variables = {
        f"gru_cell/{name}": value.detach()
        for name, value in [
            ("weight_i", module.weight_ih.reshape(3, 8, 5).transpose(1, 2)),
            ("bias_i", module.bias_ih.reshape(3, 8)),
            ("weight_h", module.weight_hh.reshape(3, 8, 8).transpose(1, 2)),
            ("bias_h", module.bias_hh.reshape(3, 8)),
        ]
    }

    # Check tensorflow impl
    with tf.Graph().as_default(), tf.Session() as session:
        output = model.gru_cell(tf.constant(hidden), tf.constant(input))
        _set_variables(session, tf_variables)
        _assert_allclose(session.run(output), ref_out.detach())


def test_transformer_conv() -> None:
    # Generate a reference output using PyTorch-geometric
    T.manual_seed(8473)
    module = torch_geometric.nn.TransformerConv(10, 20, heads=2, edge_dim=15)
    eg_n, eg_idx, eg_e = T.randn(7, 10), T.randint(0, 7,
                                                   (2, 11)), T.randn(11, 15)
    ref_out = module(eg_n, eg_idx, eg_e)

    tf_variables = {
        f"transformer_conv/{name}": T.cat(values).T.detach()
        for name, *values in [
            ("skip/linear/weight", module.lin_skip.weight),
            ("skip/linear/bias", module.lin_skip.bias),
            ("edge_shared_kv/linear/weight", module.lin_edge.weight),
            (
                "node_qkv/linear/weight",
                module.lin_query.weight,
                module.lin_key.weight,
                module.lin_value.weight,
            ),
            (
                "node_qkv/linear/bias",
                module.lin_query.bias,
                module.lin_key.bias,
                module.lin_value.bias,
            ),
        ]
    }

    # Check tensorflow impl
    with tf.Graph().as_default(), tf.Session() as session:
        output = model.transformer_conv(
            n_output=40,
            n_heads=2,
            dropout=0.0,
            nodes=tf.constant(eg_n),
            edge_idx=tf.constant(eg_idx),
            edges=tf.constant(eg_e),
        )
        _set_variables(session, tf_variables)
        _assert_allclose(session.run(output), ref_out.detach())


###############################################################################
# TGN


def test_time_encoder() -> None:
    # Generate a reference output using PyTorch-geometric
    T.manual_seed(2254)
    module = torch_geometric.nn.models.tgn.TimeEncoder(7)
    eg_input = T.rand(11)
    ref_out = module(eg_input)

    tf_variables = {
        "time_encoder/weight": module.lin.weight.view(-1).detach(),
        "time_encoder/bias": module.lin.bias.detach(),
    }

    # Check tensorflow impl
    with tf.Graph().as_default(), tf.Session() as session:
        output = model.time_encoder(tf.constant(eg_input), 7, tf.float32)
        _set_variables(session, tf_variables)
        _assert_allclose(session.run(output), ref_out.detach())


def test_tgn_memory() -> None:
    with tf.Graph().as_default(), tf.Session() as session:
        inputs = dict(
            node_ids=tf.placeholder(tf.int32, (6, )),
            write_idx=tf.placeholder(tf.int32, (2, 3)),
            write_mask=tf.placeholder(tf.bool, (2, 3)),
            write_features=tf.placeholder(tf.float32, (3, 12)),
            write_times=tf.placeholder(tf.int32, (3, )),
        )
        memory = model.tgn_memory(n_nodes=100,
                                  memory_size=14,
                                  time_embedding_size=8,
                                  **inputs)

        def step(**args: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            # Request "updates" so that they happen, but don't return them
            output, last_update, _ = session.run(
                (memory.output, memory.last_update, memory.updates),
                {
                    inputs[key]: args[key]
                    for key in args.keys() | inputs.keys()
                },
            )
            return output, last_update

        session.run(tf.global_variables_initializer())

        # Step 0, write {10, 20, 30, 40}
        output, last_update = step(
            node_ids=np.array([0, 10, 20, 30, 40, -1]),
            write_idx=np.array([[4, 1], [3, 1], [2, 3]]).T,
            write_mask=np.array([[True, False], [False, True], [True,
                                                                True]]).T,
            write_features=np.random.RandomState(100).randn(3, 12).astype(
                np.float32),
            write_times=np.array([1000, 2000, 3000]),
        )
        np.testing.assert_equal(last_update, 0)
        assert output.shape == (6, 14)
        original_output = output[0]
        np.testing.assert_allclose(output,
                                   np.tile(original_output, (6, 1)),
                                   atol=1e-6)

        # Step 1, read {0, 20, 30, 40, 50}, writes don't matter
        output, last_update = step(
            node_ids=np.array([0, 20, 30, 40, 50, -1]),
            write_idx=np.array([[0, 1], [0, 1], [5, 5]]).T,
            write_mask=np.array([[False, False], [True, True], [False,
                                                                False]]).T,
            write_features=np.zeros((3, 12), np.float32),
            write_times=np.array([4000, 5000, 6000]),
        )
        np.testing.assert_equal(last_update[:5], [0, 3000, 3000, 1000, 0])
        np.testing.assert_allclose(output[0], original_output, atol=1e-6)
        assert not np.allclose(output[1], output[0], atol=1e-6)
        # Direction doesn't matter for step 0 (since prev memory is zero)
        # assert not np.allclose(output[2], output[1], atol=1e-6)
        assert not np.allclose(output[3], output[2], atol=1e-6)
        np.testing.assert_allclose(output[4], original_output, atol=1e-6)


def test_tgn_gnn() -> None:
    # Basic "shapes and not NaN" test
    random = np.random.RandomState(9688)
    inputs = dict(
        input=random.normal(size=(7, 12)).astype(np.float32),
        last_update=random.randint(0, 100, size=7),
        edge_idx=random.randint(0, 7, size=(2, 17)),
        edge_times=random.randint(100, 200, size=17),
        edge_features=random.normal(size=(17, 8)).astype(np.float32),
    )
    with tf.Graph().as_default(), tf.Session() as session:
        out = model.tgn_gnn(
            time_embedding_size=100,
            dropout=0,
            **{k: tf.constant(v)
               for k, v in inputs.items()},
        )
        session.run(tf.global_variables_initializer())
        result = session.run(out)
        assert result.shape == (7, 12)
        assert not np.any(np.isnan(result))


def test_tgn_link_predictor() -> None:
    # Basic "shapes and not NaN" test
    random = np.random.RandomState(882)
    inputs = dict(src=random.randn(2, 7, 12), dst=random.randn(2, 7, 12))

    with tf.Graph().as_default(), tf.Session() as session:
        out = model.tgn_link_predictor(
            **{k: tf.constant(v)
               for k, v in inputs.items()})
        session.run(tf.global_variables_initializer())
        result = session.run(out)
        assert result.shape == (2, 7)
        assert not np.any(np.isnan(result))


def test_tgn() -> None:
    # Basic "shapes and not NaN" test
    random = np.random.RandomState(2691)
    n_nodes = 7
    inputs = dict(
        node_ids=random.randint(n_nodes, size=(13, ), dtype=np.int32),
        batch_idx=random.randint(13, size=(3, 4), dtype=np.int32),
        batch_times=random.randint(1000, size=4, dtype=np.int32),
        batch_features=random.randn(4, 8).astype(np.float32),
        batch_most_recent=np.ones((2, 4), np.bool),
        edge_idx=random.randint(13, size=(2, 17), dtype=np.int32),
        edge_times=random.randint(1000, size=17, dtype=np.int32),
        edge_features=random.randn(17, 6).astype(np.float32),
    )
    with tf.Graph().as_default(), tf.Session() as session:
        out = model.tgn(
            n_nodes=n_nodes,
            memory_size=12,
            time_embedding_size=14,
            dropout=0.1,
            learning_rate=1e-3,
            is_training=True,
            target="ipu",
            **{k: tf.constant(v)
               for k, v in inputs.items()},
        )
        session.run(tf.global_variables_initializer())
        result = session.run(out)
        assert 0 <= result["loss"] <= 100
        assert result["count"] <= 4
