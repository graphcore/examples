# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest
import tensorflow as tf
from ipu_tensorflow_addons.keras.layers import LayerNormalization as IpuLayerNormalization
from tensorflow.python.keras.layers.core import TFOpLambda

from model.model import AdjacencyProcessing, GcnLayer, create_model


@pytest.mark.parametrize('num_labels', [1, 2, 10])
def test_create_model_num_labels(num_labels):
    model = create_model(num_labels, 3, 4, 5, 6, 0.0, {"transform_mode": "self_connections_scaled_by_degree"})
    assert model.output.shape[-1] == num_labels


@pytest.mark.parametrize('num_features', [1, 2, 10])
def test_create_model_num_features(num_features):
    model = create_model(3, num_features, 4, 5, 6, 0.0, {"transform_mode": "self_connections_scaled_by_degree"})
    for input_tensor in model.inputs:
        if input_tensor.name == "features":
            assert input_tensor.shape[-1] == num_features


@pytest.mark.parametrize('batch_size', [4, 10])
def test_create_model_batch_size(batch_size):
    model = create_model(2, 3, batch_size, 5, 6, 0.0, {"transform_mode": "self_connections_scaled_by_degree"})
    for input_tensor in model.inputs:
        if input_tensor.name == "adjacency":
            assert input_tensor.shape == tf.TensorShape((batch_size, batch_size))
        if input_tensor.name == "features":
            assert input_tensor.shape[0] == batch_size


@pytest.mark.parametrize('hidden_size', [1, 2, 10])
def test_create_model_hidden_size(hidden_size):
    model = create_model(2, 3, 4, hidden_size, 6, 0.0, {"transform_mode": "self_connections_scaled_by_degree"})
    for layer in model.layers:
        if isinstance(layer, GcnLayer) and layer.name != "output":
            for weight in layer.transform.weights:
                assert weight.shape[-1] == hidden_size


@pytest.mark.parametrize('num_layers', [1, 2, 10])
def test_create_model_num_layers(num_layers):
    model = create_model(2, 3, 4, 5, num_layers, 0.0, {"transform_mode": "self_connections_scaled_by_degree"})
    counter = 0
    for layer in model.layers:
        if isinstance(layer, GcnLayer) or layer.name == "output":
            counter += 1
    assert counter == num_layers


@pytest.mark.parametrize('dropout_rate', [0, 0.5, 1])
def test_create_model_dropout(dropout_rate):
    model = create_model(2, 3, 4, 5, 6, dropout_rate, {"transform_mode": "self_connections_scaled_by_degree"})
    for layer in model.layers:
        if isinstance(layer, GcnLayer):
            assert layer.dropout.rate == dropout_rate


@pytest.mark.parametrize('adjacency_params', [
    {"transform_mode": "normalised_regularised", "regularisation": 0.001},
    {"transform_mode": "self_connections_scaled_by_degree"},
    {"transform_mode": "normalised_regularised_self_connections_scaled_by_degree", "regularisation": 0.001},
    {"transform_mode": "self_connections_scaled_by_degree_with_diagonal_enhancement", "diag_lambda": 0.001},
])
def test_create_model_adjacency_params(adjacency_params):
    model = create_model(2, 3, 4, 5, 6, 0.0, adjacency_params)
    for layer in model.layers:
        if isinstance(layer, AdjacencyProcessing):
            assert layer.transform_mode == adjacency_params["transform_mode"]
            assert layer.diag_lambda == adjacency_params.get("diag_lambda", None)
            assert layer.regularisation == adjacency_params.get("regularisation", None)


def test_create_model_gcn_layers():
    model = create_model(2, 3, 4, 5, 6, 0.5, {"transform_mode": "self_connections_scaled_by_degree"})
    for layer in model.layers:
        if isinstance(layer, GcnLayer):
            if layer.name != "output":
                assert len(layer.submodules) == 4
                assert isinstance(layer.layer_norm, IpuLayerNormalization)
                tf.keras.activations.serialize(layer.activation_fn) == 'relu'
            else:
                assert len(layer.submodules) == 3
                for sub_layer in layer.submodules:
                    assert not isinstance(sub_layer, IpuLayerNormalization)
                    tf.keras.activations.serialize(layer.activation_fn) == 'linear'


def expand_gcn_layer():
    model = create_model(2, 3, 4, 5, 6, 0.5, {"transform_mode": "self_connections_scaled_by_degree"})
    for layer in model.layers:
        if isinstance(layer, GcnLayer):
            gcn_layer_model = tf.keras.Model(
                model.inputs,
                layer.call(model.inputs[1], tf.cast(model.inputs[0], dtype=tf.float32))
            )
            break
    return gcn_layer_model


def test_gcn_layer():
    expected_order = ['concatenate', 'dropout', 'dense', 'layer_norm', 'tf.nn.relu']
    gcn_layer_model = expand_gcn_layer()

    order_sub_layers = list()
    for sub_layer in gcn_layer_model.layers:
        if not isinstance(sub_layer, (tf.keras.layers.InputLayer, TFOpLambda)):
            order_sub_layers.append(sub_layer.name)
    order_sub_layers.append(gcn_layer_model.layers[-1].name)

    for i in range(len(expected_order)):
        assert order_sub_layers[i].startswith(expected_order[i])
