# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from absl import flags
from tensorflow import keras

import xpu
from embedding_layers import AtomEncoder, BondEncoder

flags.DEFINE_enum("rn_multiplier", "none", ("constant", "softplus", "none"), help="RN multiplier")
flags.DEFINE_enum("decoder_mode", "node_global", ("node_global", "global", "node"), "decoder mode")
flags.DEFINE_enum(
    "mlp_norm",
    "layer_hidden",
    ("none", "layer_hidden", "layer_output"),
    "For the MLPs, whether and where to use normalization.",
)
flags.DEFINE_enum(
    "gather_scatter",
    "grouped",
    ("grouped", "debug", "dense"),
    "if `grouped`, use the batch axis to separate packs which cannot speak to each other. This may "
    "speed up computation by using grouped gather/scatter underlying implementations. "
    "If `dense`, senders/receivers will be one-hot matrices and matmuls will be used. "
    "If `debug`, will use a list comprehension over the batch dimension (this is bad and slow "
    "but may be useful for debugging",
)

FLAGS = flags.FLAGS


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, edge_model_fn, node_model_fn):
        super().__init__()
        self.atom_encoder = AtomEncoder(FLAGS.n_embedding_channels)
        self.bond_encoder = BondEncoder(FLAGS.n_embedding_channels)

        # instantiates the MLPs here
        self.edge_model = edge_model_fn()
        self.node_model = node_model_fn()
        self.global_latent = keras.layers.Embedding(1, int(FLAGS.n_latent))

    def call(self, inputs, training=False):
        nodes, edges, receivers, senders, node_graph_idx, edge_graph_idx = inputs

        nodes = self.atom_encoder(nodes)
        edges = self.bond_encoder(edges)

        edges_update = self.edge_model(edges)
        nodes_update = self.node_model(nodes)
        # each graph has the same learned global embedding
        global_latent = self.global_latent(
            tf.constant([0] * FLAGS.micro_batch_size * FLAGS.n_graphs_per_pack, dtype=tf.int32)
        )
        global_latent = tf.reshape(global_latent, [FLAGS.micro_batch_size, FLAGS.n_graphs_per_pack, -1])

        return nodes_update, edges_update, receivers, senders, global_latent, node_graph_idx, edge_graph_idx


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, global_model_fn=None):
        super().__init__()
        # instantiates the MLPs here
        self.global_model = global_model_fn()

    def call(self, inputs, training=False):
        nodes, edges, receivers, senders, global_latent, node_graph_idx, edge_graph_idx = inputs
        if FLAGS.decoder_mode == "node_global":
            node_inputs_to_global_decoder = _scatter(
                nodes, node_graph_idx, num_segments=FLAGS.n_graphs_per_pack, gather_scatter_method=FLAGS.gather_scatter
            )
            inputs_to_decoder = tf.concat([node_inputs_to_global_decoder, global_latent], axis=-1)
        elif FLAGS.decoder_mode == "node":
            inputs_to_decoder = _scatter(
                nodes, node_graph_idx, num_segments=FLAGS.n_graphs_per_pack, gather_scatter_method=FLAGS.gather_scatter
            )
        elif FLAGS.decoder_mode == "global":
            inputs_to_decoder = global_latent
        else:
            raise ValueError("Pick a relevant decoder mode")
        logits = self.global_model(inputs_to_decoder)
        return logits


class GraphNetworkLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        edge_model_fn,
        node_model_fn,
        global_model_fn,
        use_residual=True,
        is_last_layer=False,
        nodes_dropout=0.0,
        edges_dropout=0.0,
        globals_dropout=0.0,
    ):
        super().__init__()
        # Instantiates the MLPs here
        self.edge_model = edge_model_fn()
        self.node_model = node_model_fn()
        self.global_model = global_model_fn()
        self.use_residual = use_residual
        self.is_last_layer = is_last_layer
        self.node_dropout = xpu.Dropout(
            rate=nodes_dropout, noise_shape=(FLAGS.micro_batch_size, FLAGS.n_nodes_per_pack, 1)
        )
        self.edge_dropout = xpu.Dropout(
            rate=edges_dropout, noise_shape=(FLAGS.micro_batch_size, FLAGS.n_edges_per_pack, 1)
        )
        self.global_dropout = xpu.Dropout(
            rate=globals_dropout, noise_shape=(FLAGS.micro_batch_size, FLAGS.n_graphs_per_pack, 1)
        )
        if FLAGS.rn_multiplier in ("constant", "softplus"):
            self.residual_multiplier = ResnetMultiplier(method=FLAGS.rn_multiplier)
        else:
            self.residual_multiplier = lambda x: x

    def call(self, inputs, training=False):
        nodes, edges, receivers, senders, global_latent, node_graph_idx, edge_graph_idx = inputs
        # ---------------EDGE step---------------
        # nodes mapped to edges
        received_attributes = _gather(nodes, receivers, gather_scatter_method=FLAGS.gather_scatter)
        sent_attributes = _gather(nodes, senders, gather_scatter_method=FLAGS.gather_scatter)
        global_latent_edge = _gather(global_latent, edge_graph_idx, gather_scatter_method=FLAGS.gather_scatter)
        # global_latent affects each node the same
        edges_update = tf.concat([edges, sent_attributes, received_attributes, global_latent_edge], axis=-1)
        edges_update = self.edge_model(edges_update)

        # ---------------NODE step---------------
        adjacent_edges = _scatter(
            edges_update, receivers, num_segments=FLAGS.n_nodes_per_pack, gather_scatter_method=FLAGS.gather_scatter
        )
        node_shaped_global = _gather(global_latent, node_graph_idx, gather_scatter_method=FLAGS.gather_scatter)
        nodes_update = tf.concat([nodes, adjacent_edges, node_shaped_global], axis=-1)
        nodes_update = self.node_model(nodes_update)

        # ---------------GLOBAL step---------------
        node_aggregate = _scatter(
            nodes_update,
            node_graph_idx,
            num_segments=FLAGS.n_graphs_per_pack,
            gather_scatter_method=FLAGS.gather_scatter,
        )
        edge_aggregate = _scatter(
            edges_update,
            edge_graph_idx,
            num_segments=FLAGS.n_graphs_per_pack,
            gather_scatter_method=FLAGS.gather_scatter,
        )
        global_latent_update = tf.concat([node_aggregate, edge_aggregate, global_latent], -1)
        global_latent_update = self.global_model(global_latent_update)

        # dropout before the residual block`
        nodes_update = self.node_dropout(nodes_update, training=training)
        edges_update = self.edge_dropout(edges_update, training=training)
        global_latent_update = self.global_dropout(global_latent_update, training=training)

        # 'residual' always on
        edges_update = edges + self.residual_multiplier(edges_update)
        nodes_update = nodes + self.residual_multiplier(nodes_update)
        global_latent_update = global_latent + self.residual_multiplier(global_latent_update)
        return nodes_update, edges_update, receivers, senders, global_latent_update, node_graph_idx, edge_graph_idx


class InteractionNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, edge_model_fn, node_model_fn, nodes_dropout=0.0, edges_dropout=0.0):
        super().__init__()
        # instantiates the MLPs here
        self.edge_model = edge_model_fn()
        self.node_model = node_model_fn()
        self.node_dropout = xpu.Dropout(
            rate=nodes_dropout, noise_shape=(FLAGS.micro_batch_size, FLAGS.n_nodes_per_pack, 1)
        )
        self.edge_dropout = xpu.Dropout(
            rate=edges_dropout, noise_shape=(FLAGS.micro_batch_size, FLAGS.n_edges_per_pack, 1)
        )
        if FLAGS.rn_multiplier in ("constant", "softplus"):
            self.residual_multiplier = ResnetMultiplier(method=FLAGS.rn_multiplier)
        else:
            self.residual_multiplier = lambda x: x

    def call(self, inputs, training=False):
        nodes, edges, receivers, senders, global_latent, node_graph_idx, edge_graph_idx = inputs
        # ---------------EDGE step---------------
        # nodes mapped to edges
        received_attributes = _gather(nodes, receivers, gather_scatter_method=FLAGS.gather_scatter)
        sent_attributes = _gather(nodes, senders, gather_scatter_method=FLAGS.gather_scatter)
        edges_update = tf.concat([edges, sent_attributes, received_attributes], axis=-1)
        edges_update = self.edge_model(edges_update)

        # ---------------NODE step---------------
        adjacent_edges = _scatter(
            edges_update, receivers, num_segments=FLAGS.n_nodes_per_pack, gather_scatter_method=FLAGS.gather_scatter
        )
        nodes_update = tf.concat([nodes, adjacent_edges], axis=-1)
        nodes_update = self.node_model(nodes_update)

        # dropout before the residual block`
        nodes_update = self.node_dropout(nodes_update, training=training)
        edges_update = self.edge_dropout(edges_update, training=training)

        # 'residual' always on
        edges_update = edges + self.residual_multiplier(edges_update)
        nodes_update = nodes + self.residual_multiplier(nodes_update)

        return nodes_update, edges_update, receivers, senders, global_latent, node_graph_idx, edge_graph_idx


class GinEncoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.atom_encoder = AtomEncoder(FLAGS.n_embedding_channels)

    def call(self, inputs, training=False):
        nodes, edges, receivers, senders, node_graph_idx, edge_graph_idx = inputs

        nodes = self.atom_encoder(nodes)
        return (
            nodes,
            edges,
            receivers,
            senders,
            tf.zeros([FLAGS.micro_batch_size, FLAGS.n_graphs_per_pack, 1]),
            node_graph_idx,
            edge_graph_idx,
        )


class GinDecoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        nodes, edges, receivers, senders, global_latent, node_graph_idx, edge_graph_idx = inputs
        graphwise_means = _batched_segment_mean(
            nodes, node_graph_idx, graphs_per_pack=FLAGS.n_graphs_per_pack, gather_scatter_method=FLAGS.gather_scatter
        )

        output_prob = self.output_layer(graphwise_means)
        return output_prob


class GraphIsomorphismLayer(tf.keras.layers.Layer):
    def __init__(self, get_mlp, use_edges=True, epsilon=None, edge_dim=100, dropout=0.0):
        """

        :param get_mlp:
        :param use_edges: if True, generate per-layer embeddings for the edges
        :param epsilon:
        """
        super().__init__()
        self.mlp = get_mlp()
        self.use_edges = use_edges
        self.edge_encoder = BondEncoder(int(edge_dim)) if use_edges else None
        if dropout > 0:
            self.dropout = xpu.Dropout(rate=dropout, noise_shape=(FLAGS.micro_batch_size, FLAGS.n_nodes_per_pack, 1))
        else:
            self.dropout = None
        self.epsilon = epsilon

    def build(self, input_shape):
        super().build(input_shape)
        if self.epsilon is None:
            self.epsilon = self.add_weight(shape=(1,), initializer="zeros", name="epsilon")
        else:
            self.epsilon = tf.cast(self.epsilon, self.dtype)

    def call(self, inputs, training=False):
        nodes, edges, receivers, senders, global_latent, node_graph_idx, edge_graph_idx = inputs
        sent_attributes = _gather(nodes, senders, gather_scatter_method=FLAGS.gather_scatter)

        if self.use_edges:
            sent_attributes += self.edge_encoder(edges)

        message = _scatter(
            sent_attributes, receivers, num_segments=FLAGS.n_nodes_per_pack, gather_scatter_method=FLAGS.gather_scatter
        )

        node_update = self.mlp((1 + self.epsilon) * nodes + message)
        if self.dropout is not None:
            node_update = self.dropout(node_update)
        return node_update, edges, receivers, senders, global_latent, node_graph_idx, edge_graph_idx


class MLP(keras.layers.Layer):
    def __init__(self, n_layers, n_hidden, n_out, dense_kernel_regularization=0.0, activate_final=False, name="MLP"):
        super().__init__(name=name)
        self.mlp = tf.keras.models.Sequential()
        for _ in range(n_layers - 1):
            self.mlp.add(
                keras.layers.Dense(
                    n_hidden, activation=None, kernel_regularizer=keras.regularizers.l2(dense_kernel_regularization)
                )
            )
            # norm PRIOR to the activation
            if FLAGS.mlp_norm == "layer_hidden":
                self.mlp.add(xpu.LayerNormalization())

            self.mlp.add(tf.keras.layers.Activation("relu"))

        self.mlp.add(
            keras.layers.Dense(
                n_out, activation=None, kernel_regularizer=keras.regularizers.l2(dense_kernel_regularization)
            )
        )

        if FLAGS.mlp_norm == "layer_output":
            self.mlp.add(xpu.LayerNormalization())
        if activate_final:
            self.mlp.add(tf.keras.layers.Activation("relu"))

    def call(self, inputs, training=False):
        return self.mlp(inputs)


class ResnetMultiplier(keras.layers.Layer):
    def __init__(self, method="constant", name="MLP"):
        super().__init__(name=name)
        self.method = method

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if self.method == "constant":
            init_val = 0.0
        elif self.method == "softplus":
            init_val = -5.0
        else:
            raise NotImplementedError()

        init = tf.constant_initializer(init_val)
        self.w = self.add_weight("value", shape=[1], initializer=init, dtype=dtype)
        self.built = True

    def call(self, inputs, _training=True):
        if self.method == "constant":
            return self.w * inputs
        elif self.method == "softplus":
            return tf.nn.softplus(self.w) * inputs
        else:
            raise NotImplementedError("Invalid method for resnet multiplier")


def _gather(data, indices, gather_scatter_method="grouped"):
    if gather_scatter_method == "grouped":
        x = xpu.grouped_gather(data, indices)
    elif gather_scatter_method == "dense":
        x = tf.matmul(tf.cast(indices, data.dtype), data)
    elif gather_scatter_method == "debug":
        x = tf.stack([tf.gather(data[i], indices[i]) for i in range(data.shape[0])])
    else:
        raise ValueError(f"gather_scatter method {gather_scatter_method} is invalid.")
    return x


def _scatter(data, indices, num_segments, gather_scatter_method="grouped"):
    if gather_scatter_method == "grouped":
        x = xpu.grouped_scatter(data, indices, table_size=num_segments)
    elif gather_scatter_method == "dense":
        # here, senders and receivers are dense matrices
        x = tf.matmul(tf.cast(indices, data.dtype), data, transpose_a=True)
    elif gather_scatter_method == "debug":
        unbatched_xs = [
            tf.math.unsorted_segment_sum(data[i], indices[i], num_segments=num_segments) for i in range(data.shape[0])
        ]
        x = tf.stack(unbatched_xs)
    else:
        raise ValueError(f"gather_scatter method {gather_scatter_method} is invalid.")
    return x


def _batched_segment_mean(x, indices, graphs_per_pack, gather_scatter_method="debug"):
    # the number of nodes per graph COULD be computed in the data loader, if desired
    if gather_scatter_method == "grouped":
        graphwise_means = xpu.grouped_scatter(x, indices, table_size=graphs_per_pack)
        # we need to count the nodes per graph
        dummy_nodes = tf.ones_like(indices, dtype=x.dtype)
        n_nodes_per_graph = xpu.grouped_scatter(dummy_nodes[..., tf.newaxis], indices, table_size=graphs_per_pack)
        # where there are no nodes, divide by 1 not 0
        graphwise_means /= tf.maximum(n_nodes_per_graph, tf.constant(1, x.dtype))
    elif gather_scatter_method == "dense":
        # shape: (batch, graphs/pack, n_hidden)
        indices = tf.cast(indices, x.dtype)
        graphwise_means = tf.matmul(indices, x, transpose_a=True)
        # the last dimension is the hidden dimension. The previous is the one we should reduce over
        n_nodes_per_graph = tf.reduce_sum(indices, axis=-2)[..., tf.newaxis]
        graphwise_means /= tf.maximum(n_nodes_per_graph, tf.constant(1, x.dtype))
    elif gather_scatter_method == "debug":
        graphwise_means = [
            tf.math.unsorted_segment_mean(x[i], indices[i], num_segments=graphs_per_pack) for i in range(x.shape[0])
        ]
        graphwise_means = tf.stack(graphwise_means)
    else:
        raise ValueError(f"gather_scatter method {gather_scatter_method} is invalid.")

    return graphwise_means
