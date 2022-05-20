# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from ipu_tensorflow_addons.keras.layers import Dropout as IpuDropout
from ipu_tensorflow_addons.keras.layers import LayerNormalization as IpuLayerNormalization

from model.adjacency_processing import AdjacencyProcessing


class GcnLayer(tf.keras.layers.Layer):
    """
    Graph Convolution Network layer that performs the following operations on its input:
    1. Dot product with adjacency matrix: AX.
    2. Concatenation: [AX, X].
    3. Dropout and the corresponding scaling of each component of [AX, X].
    4. Dot product with the weights and add bias.
    5. Layer normalise the result. Layer normalisation is used for its better performance on the IPU, but it is
        equivalent to batch normalisation for 2D inputs.
    6. Activation function.
    """
    def __init__(
            self,
            dropout_rate,
            out_dim,
            activation_fn=tf.keras.activations.relu,
            do_norm=True,
            first_layer_precalculation=False,
            use_ipu_layers=True,
            **args
    ):
        """
        Graph Convolution Network layer.
        Args:
          dropout_rate: rate: Float between 0 and 1. Fraction of the input units to drop.
          out_dim: Integer with the output dimension. Number of columns of the weight matrix.
          activation_fn: Activation function at the output of the layer.
          do_norm: Boolean flag indicating whether applying normalisation or not.
        """
        super().__init__(**args)
        self.concatenate = tf.keras.layers.Concatenate()

        dropout_class = IpuDropout if use_ipu_layers else tf.keras.layers.Dropout
        self.dropout = dropout_class(dropout_rate)

        layer_norm_class = IpuLayerNormalization if use_ipu_layers else tf.keras.layers.LayerNormalization
        self.layer_norm = layer_norm_class(epsilon=1e-09) if do_norm else lambda x: x

        self.transform = tf.keras.layers.Dense(out_dim, use_bias=False)
        self.activation_fn = activation_fn
        self.first_layer_precalculation = first_layer_precalculation

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, adjacency):
        if self.first_layer_precalculation:
            ax_x = inputs
        else:
            ax = (adjacency @ inputs)
            ax_x = self.concatenate([ax, inputs])
        ax_x_drop = self.dropout(ax_x)
        transformed = self.transform(ax_x_drop)
        transformed_norm = self.layer_norm(transformed)
        features = self.activation_fn(transformed_norm)
        return features


def create_model(
        num_labels,
        num_features,
        max_nodes_per_batch,
        hidden_size,
        num_layers,
        dropout_rate,
        adjacency_params,
        cast_model_inputs_to_dtype=tf.float32,
        first_layer_precalculation=False,
        use_ipu_layers=True
):
    """Create a GCN model."""
    adjacency = tf.keras.Input(max_nodes_per_batch,
                               batch_size=max_nodes_per_batch,
                               dtype=tf.bool,
                               name="adjacency")
    features = tf.keras.Input(num_features,
                              batch_size=max_nodes_per_batch,
                              dtype=cast_model_inputs_to_dtype,
                              name="features")

    # Add adjacency matrix preprocessing
    adjacency_matrix_layer = AdjacencyProcessing(
        **adjacency_params,
        name="adjacency_processing"
    )
    gnn_adjacency = adjacency_matrix_layer(tf.cast(adjacency, dtype=cast_model_inputs_to_dtype))
    hidden = features

    if num_layers > 1:
        hidden = GcnLayer(
            dropout_rate,
            hidden_size,
            name=f"gcn_0",
            first_layer_precalculation=first_layer_precalculation,
            use_ipu_layers=use_ipu_layers
        )(hidden, gnn_adjacency)

        # Hidden layers
        for n in range(1, num_layers - 1):
            hidden = GcnLayer(
                dropout_rate,
                hidden_size,
                name=f"gcn_{n}",
                use_ipu_layers=use_ipu_layers
            )(hidden, gnn_adjacency)

    # Predict
    output = GcnLayer(
        dropout_rate=dropout_rate,
        out_dim=num_labels,
        activation_fn=tf.keras.activations.linear,
        do_norm=False,
        name="output",
        use_ipu_layers=use_ipu_layers
    )(hidden, gnn_adjacency)

    return tf.keras.Model(dict(adjacency=adjacency, features=features), output)
