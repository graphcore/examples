# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest
import tensorflow as tf
from ipu_tensorflow_addons.keras.layers import LayerNormalization as IpuLayerNormalization
from keras.layers.core import TFOpLambda
from keras.engine.keras_tensor import SparseKerasTensor

from model.model import AdjacencyProcessing, GcnLayer, create_model
from utilities.constants import AdjacencyForm


@pytest.mark.parametrize('num_labels', [1, 2, 10])
def test_create_model_num_labels(num_labels):
    model = create_model(1, num_labels, 3, 4, 10, 5, 6, 0.0, {"transform_mode": "self_connections_scaled_by_degree"})
    assert model.output.shape[-1] == num_labels


@pytest.mark.parametrize('num_features', [1, 2, 10])
def test_create_model_num_features(num_features):
    model = create_model(1, 3, num_features, 4, 10, 5, 6, 0.0, {"transform_mode": "self_connections_scaled_by_degree"})
    for input_tensor in model.inputs:
        if input_tensor.name == "features":
            assert input_tensor.shape[-1] == num_features


@pytest.mark.parametrize('num_nodes', [4, 10])
def test_create_model_num_nodes(num_nodes):
    model = create_model(1, 2, 3, num_nodes, 10, 5, 6, 0.0, {"transform_mode": "self_connections_scaled_by_degree"})
    for input_tensor in model.inputs:
        if input_tensor.name == "adjacency":
            assert input_tensor.shape == tf.TensorShape((num_nodes, num_nodes))
        if input_tensor.name == "features":
            assert input_tensor.shape[0] == num_nodes


@pytest.mark.parametrize('max_num_edges', [10, 20])
@pytest.mark.parametrize('adjacency_form', [AdjacencyForm.DENSE,
                                            AdjacencyForm.SPARSE_TENSOR,
                                            AdjacencyForm.SPARSE_TUPLE])
def test_create_model_num_edges(max_num_edges, adjacency_form):
    num_nodes = 4
    model = create_model(1, 2, 3, num_nodes, max_num_edges, 5, 6, 0.0,
                         {"transform_mode": "self_connections_scaled_by_degree"},
                         adjacency_form=adjacency_form)
    for input_var in model.inputs:
        if input_var.name == "adjacency":
            if adjacency_form == AdjacencyForm.DENSE:
                assert input_var.shape == tf.TensorShape((num_nodes, num_nodes))
            elif adjacency_form == AdjacencyForm.SPARSE_TENSOR:
                assert isinstance(input_var, SparseKerasTensor)
                assert input_var.shape == tf.TensorShape((num_nodes, num_nodes))
            else:
                assert isinstance(input_var, tuple)
                assert len(input_var) == 3
                assert input_var[0].name == "adjacency_edges"
                assert input_var[0].shape == tf.TensorShape((max_num_edges, 2))
                assert input_var[1].name == "adjacency_values"
                assert input_var[1].shape == tf.TensorShape((max_num_edges,))
                assert input_var[2]


@pytest.mark.parametrize('hidden_size', [1, 2, 10])
@pytest.mark.parametrize('adjacency_form', [AdjacencyForm.DENSE,
                                            AdjacencyForm.SPARSE_TENSOR,
                                            AdjacencyForm.SPARSE_TUPLE])
def test_create_model_hidden_size(hidden_size, adjacency_form):
    model = create_model(1, 2, 3, 4, 10, hidden_size, 6, 0.0,
                         {"transform_mode": "self_connections_scaled_by_degree"},
                         adjacency_form=adjacency_form)
    for layer in model.layers:
        if isinstance(layer, GcnLayer) and layer.name != "output":
            for weight in layer.transform.weights:
                assert weight.shape[-1] == hidden_size


@pytest.mark.parametrize('num_layers', [1, 2, 10])
@pytest.mark.parametrize('adjacency_form', [AdjacencyForm.DENSE,
                                            AdjacencyForm.SPARSE_TENSOR,
                                            AdjacencyForm.SPARSE_TUPLE])
def test_create_model_num_layers(num_layers, adjacency_form):
    model = create_model(1, 2, 3, 4, 10, 5, num_layers, 0.0,
                         {"transform_mode": "self_connections_scaled_by_degree"},
                         adjacency_form=adjacency_form)
    counter = 0
    for layer in model.layers:
        if isinstance(layer, GcnLayer) or layer.name == "output":
            counter += 1
    assert counter == num_layers


@pytest.mark.parametrize('dropout_rate', [0, 0.5, 1])
@pytest.mark.parametrize('adjacency_form', [AdjacencyForm.DENSE,
                                            AdjacencyForm.SPARSE_TENSOR,
                                            AdjacencyForm.SPARSE_TUPLE])
def test_create_model_dropout(dropout_rate, adjacency_form):
    model = create_model(1, 2, 3, 4, 10, 5, 6, dropout_rate,
                         {"transform_mode": "self_connections_scaled_by_degree"},
                         adjacency_form=adjacency_form)
    for layer in model.layers:
        if isinstance(layer, GcnLayer):
            assert layer.dropout.rate == dropout_rate


@pytest.mark.parametrize('adjacency_params', [
    {"transform_mode": "normalised_regularised", "regularisation": 0.001},
    {"transform_mode": "self_connections_scaled_by_degree"},
    {"transform_mode": "normalised_regularised_self_connections_scaled_by_degree", "regularisation": 0.001},
    {"transform_mode": "self_connections_scaled_by_degree_with_diagonal_enhancement", "diag_lambda": 0.001},
])
@pytest.mark.parametrize('adjacency_form', [AdjacencyForm.DENSE,
                                            AdjacencyForm.SPARSE_TENSOR,
                                            AdjacencyForm.SPARSE_TUPLE])
def test_create_model_adjacency_params(adjacency_params, adjacency_form):
    model = create_model(1, 2, 3, 4, 10, 5, 6, 0.0, adjacency_params, adjacency_form=adjacency_form)
    for layer in model.layers:
        if isinstance(layer, AdjacencyProcessing):
            assert layer.transform_mode == adjacency_params["transform_mode"]
            assert layer.diag_lambda == adjacency_params.get("diag_lambda", None)
            assert layer.regularisation == adjacency_params.get("regularisation", None)


@pytest.mark.parametrize('adjacency_form', [AdjacencyForm.DENSE,
                                            AdjacencyForm.SPARSE_TENSOR,
                                            AdjacencyForm.SPARSE_TUPLE])
def test_create_model_gcn_layers(adjacency_form):
    model = create_model(1, 2, 3, 4, 10, 5, 6, 0.5,
                         {"transform_mode": "self_connections_scaled_by_degree"},
                         adjacency_form=adjacency_form)
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


def expand_gcn_layer(adjacency_form):
    num_nodes = 4
    model = create_model(1, 2, 3, num_nodes, 10, 5, 6, 0.5,
                         {"transform_mode": "self_connections_scaled_by_degree"},
                         adjacency_form=adjacency_form)
    for layer in model.layers:
        if isinstance(layer, GcnLayer):

            # Arrange inputs to emulate the previous steps.
            if adjacency_form == AdjacencyForm.DENSE:
                adjacency, features = model.inputs
                adjacency = tf.cast(adjacency, dtype=tf.float32)
                layer_inputs = (features, adjacency)
            elif adjacency_form == AdjacencyForm.SPARSE_TENSOR:
                adjacency, features = model.inputs
                layer_inputs = (features, adjacency)
            elif adjacency_form == AdjacencyForm.SPARSE_TUPLE:
                adjacency_edges, adjacency_values, features = model.inputs
                adjacency_values = tf.cast(adjacency_values, dtype=tf.float32)
                adjacency_edges = tf.squeeze(adjacency_edges)
                adjacency_values = tf.squeeze(adjacency_values)
                adjacency_shape = tf.TensorShape((num_nodes, num_nodes))
                features = tf.squeeze(features)
                layer_inputs = (features, (adjacency_edges, adjacency_values, adjacency_shape))

            gcn_layer_model = tf.keras.Model(model.inputs, layer.call(*layer_inputs))
            break
    return gcn_layer_model


@pytest.mark.parametrize('adjacency_form', [AdjacencyForm.DENSE,
                                            AdjacencyForm.SPARSE_TENSOR,
                                            AdjacencyForm.SPARSE_TUPLE])
def test_gcn_layer(adjacency_form):
    expected_order = ['sparse_dense_matmul',
                      'concatenate',
                      'dropout',
                      'dense',
                      'layer_norm',
                      'tf.nn.relu']
    gcn_layer_model = expand_gcn_layer(adjacency_form)

    order_sub_layers = list()
    for sub_layer in gcn_layer_model.layers:
        if not isinstance(sub_layer, (tf.keras.layers.InputLayer, TFOpLambda)):
            order_sub_layers.append(sub_layer.name)
    order_sub_layers.append(gcn_layer_model.layers[-1].name)

    for i in range(len(expected_order)):
        assert order_sub_layers[i].startswith(expected_order[i])
