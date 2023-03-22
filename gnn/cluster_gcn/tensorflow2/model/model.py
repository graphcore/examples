# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow.python.distribute import distribution_strategy_context
from utilities.constants import AdjacencyForm

from model.adjacency_processing import AdjacencyProcessing
from model.gcn_layer import GcnLayer


def define_inputs_with_tensor_adjacency(
    micro_batch_size,
    max_nodes_per_batch,
    adjacency_form,
    inputs_dtype,
    num_features,
):
    assert micro_batch_size == 1, (
        f"A micro_batch_size of {micro_batch_size} has been provided,"
        " but only a micro_batch_size of 1 is currently supported."
    )

    strategy = distribution_strategy_context.get_strategy()

    adj_dtype = inputs_dtype if adjacency_form == AdjacencyForm.SPARSE_TENSOR else tf.bool
    adjacency = tf.keras.Input(
        max_nodes_per_batch,
        batch_size=max_nodes_per_batch * strategy.num_replicas_in_sync,
        dtype=adj_dtype,
        sparse=adjacency_form == AdjacencyForm.SPARSE_TENSOR,
        name="adjacency",
    )
    features = tf.keras.Input(
        num_features,
        batch_size=max_nodes_per_batch * strategy.num_replicas_in_sync,
        dtype=inputs_dtype,
        name="features",
    )
    return adjacency, features


def define_inputs_with_tuple_adjacency(
    micro_batch_size, max_edges_per_batch, inputs_dtype, num_features, max_nodes_per_batch
):
    strategy = distribution_strategy_context.get_strategy()

    adjacency_edges = tf.keras.Input(
        shape=(max_edges_per_batch, 2),
        batch_size=micro_batch_size * strategy.num_replicas_in_sync,
        dtype=tf.int32,
        name="adjacency_edges",
    )
    adjacency_values = tf.keras.Input(
        shape=(max_edges_per_batch,),
        batch_size=micro_batch_size * strategy.num_replicas_in_sync,
        dtype=inputs_dtype,
        name="adjacency_values",
    )
    adjacency = (adjacency_edges, adjacency_values)

    features = tf.keras.Input(
        (max_nodes_per_batch, num_features),
        batch_size=micro_batch_size * strategy.num_replicas_in_sync,
        dtype=inputs_dtype,
        name="features",
    )

    return adjacency, features


def squeeze_batch_dim(adjacency, features):
    """Squeeze artefact batch dimension when working with sparse tuple."""
    if isinstance(adjacency, tuple):
        adjacency_batch = tuple(tf.squeeze(a) for a in adjacency)
        features_batch = tf.squeeze(features)
        return adjacency_batch, features_batch
    else:
        return adjacency, features


def create_model(
    micro_batch_size,
    num_labels,
    num_features,
    max_nodes_per_batch,
    max_edges_per_batch,
    hidden_size,
    num_layers,
    dropout_rate,
    adjacency_params,
    cast_model_inputs_to_dtype=tf.float32,
    first_layer_precalculation=False,
    use_ipu_layers=True,
    adjacency_form=AdjacencyForm.DENSE,
):
    """Create a GCN model."""

    if adjacency_form in [AdjacencyForm.DENSE, AdjacencyForm.SPARSE_TENSOR]:
        adjacency_input, features_input = define_inputs_with_tensor_adjacency(
            micro_batch_size, max_nodes_per_batch, adjacency_form, cast_model_inputs_to_dtype, num_features
        )
    elif adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        adjacency_input, features_input = define_inputs_with_tuple_adjacency(
            micro_batch_size, max_edges_per_batch, cast_model_inputs_to_dtype, num_features, max_nodes_per_batch
        )

    adjacency, hidden = squeeze_batch_dim(adjacency_input, features_input)

    # Add adjacency matrix preprocessing layer.
    adjacency_processing_layer = AdjacencyProcessing(
        max_nodes_per_batch,
        **adjacency_params,
        adjacency_form=adjacency_form,
        adjacency_dtype=cast_model_inputs_to_dtype,
        name="adjacency_processing",
    )
    transformed_adjacency = adjacency_processing_layer(adjacency)

    # Add GCN layers.
    if num_layers > 1:
        hidden = GcnLayer(
            dropout_rate,
            hidden_size,
            name=f"gcn_0",
            first_layer_precalculation=first_layer_precalculation,
            use_ipu_layers=use_ipu_layers,
        )(hidden, transformed_adjacency)

        # Hidden layers
        for n in range(1, num_layers - 1):
            hidden = GcnLayer(
                dropout_rate,
                hidden_size,
                name=f"gcn_{n}",
                use_ipu_layers=use_ipu_layers,
            )(hidden, transformed_adjacency)

    # Predict
    output = GcnLayer(
        dropout_rate=dropout_rate,
        out_dim=num_labels,
        activation_fn=tf.keras.activations.linear,
        do_norm=False,
        name="output",
        use_ipu_layers=use_ipu_layers,
    )(hidden, transformed_adjacency)

    return tf.keras.Model(dict(adjacency_batch=adjacency_input, features_batch=features_input), output)
