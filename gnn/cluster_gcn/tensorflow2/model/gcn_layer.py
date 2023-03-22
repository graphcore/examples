# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow.keras.layers import Dropout, LayerNormalization

from ipu_tensorflow_addons.keras.layers import Dropout as IpuDropout
from ipu_tensorflow_addons.keras.layers import LayerNormalization as IpuLayerNormalization

import utilities.sparse_mat_ops as mat_ops


class GcnLayer(tf.keras.layers.Layer):
    """
    Graph Convolution Network layer that performs the following operations on its input:
    1. Dot product with adjacency matrix: AX.
    2. Concatenation: [AX, X].
    3. Dropout and the corresponding scaling of each component of [AX, X].
    4. Dot product with the weights and add bias.
    5. Layer normalise the result. Layer normalisation is used for its better performance
        on the IPU, but it is equivalent to batch normalisation for 2D inputs.
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
          first_layer_precalculation: Boolean flag indicating whether the input features to the
            first layer already include the product with the adjacency and its concatenation
            (i.e., [AX, X]).
          use_ipu_layers: Boolean flag indicating whether using IPU specific layers or standard
            upstream tf.keras layers.
        """
        super().__init__(**args)

        self.concatenate = tf.keras.layers.Concatenate()

        dropout_class = IpuDropout if use_ipu_layers else Dropout
        self.dropout = dropout_class(dropout_rate)

        layer_norm_class = IpuLayerNormalization if use_ipu_layers else LayerNormalization
        self.layer_norm = layer_norm_class(epsilon=1e-09) if do_norm else lambda x: x

        self.transform = tf.keras.layers.Dense(out_dim, use_bias=False)
        self.activation_fn = activation_fn
        self.first_layer_precalculation = first_layer_precalculation

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, features, adjacency):
        if self.first_layer_precalculation:
            ax_x = features
        else:
            ax = tf.keras.layers.Lambda(lambda x: mat_ops.sp_dense_matmul(x[0], x[1]), name="sparse_dense_matmul")(
                (adjacency, features)
            )
            ax_x = self.concatenate([ax, features])

        ax_x_drop = self.dropout(ax_x)
        transformed = self.transform(ax_x_drop)
        transformed_norm = self.layer_norm(transformed)
        new_features = self.activation_fn(transformed_norm)
        return new_features
