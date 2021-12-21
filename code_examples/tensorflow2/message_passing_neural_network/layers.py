# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import xpu
from tensorflow import keras
from embedding_layers import AtomEncoder, BondEncoder
from absl import flags

flags.DEFINE_enum('rn_multiplier', 'none', ('constant', 'softplus', 'none'), help="RN multiplier")
flags.DEFINE_enum("decoder_mode", "node_global", ("node_global", "global", "node"), "decoder mode")
flags.DEFINE_enum('mlp_norm', 'layer_hidden', ('none', 'layer_hidden', 'layer_output'),
                  "For the MLPs, whether and where to use normalization.")

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
        global_latent = self.global_latent(tf.constant([0] * FLAGS.batch_size, dtype=tf.int32))
        return nodes_update, edges_update, receivers, senders, global_latent, node_graph_idx, edge_graph_idx


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, global_model_fn=None):
        super().__init__()
        # instantiates the MLPs here
        self.global_model = global_model_fn()

    def call(self, inputs, training=False):
        nodes, edges, receivers, senders, global_latent, node_graph_idx, edge_graph_idx = inputs
        if FLAGS.decoder_mode == 'node_global':
            node_inputs_to_global_decoder = tf.math.unsorted_segment_sum(
                nodes, node_graph_idx, num_segments=global_latent.shape[0])
            inputs_to_decoder = tf.concat([node_inputs_to_global_decoder, global_latent], axis=1)
        elif FLAGS.decoder_mode == 'node':
            inputs_to_decoder = tf.math.unsorted_segment_sum(
                nodes, node_graph_idx, num_segments=global_latent.shape[0])
        elif FLAGS.decoder_mode == 'global':
            inputs_to_decoder = global_latent
        else:
            raise ValueError("pick a relevant decoder mode")
        logits = self.global_model(inputs_to_decoder)
        return logits


class GraphNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, edge_model_fn, node_model_fn, global_model_fn, use_residual=True,
                 is_last_layer=False, nodes_dropout=0., edges_dropout=0., globals_dropout=0.):
        super().__init__()
        # instantiates the MLPs here
        self.edge_model = edge_model_fn()
        self.node_model = node_model_fn()
        self.global_model = global_model_fn()
        self.use_residual = use_residual
        self.is_last_layer = is_last_layer
        self.gather = xpu.gather
        self.node_dropout = xpu.Dropout(rate=nodes_dropout, noise_shape=(FLAGS.batch_size * FLAGS.n_nodes, 1))
        self.edge_dropout = xpu.Dropout(rate=edges_dropout, noise_shape=(FLAGS.batch_size * FLAGS.n_edges, 1))
        self.global_dropout = xpu.Dropout(rate=globals_dropout, noise_shape=(FLAGS.batch_size, 1))
        if FLAGS.rn_multiplier in ('constant', 'softplus'):
            self.residual_multiplier = ResnetMultiplier(method=FLAGS.rn_multiplier)
        else:
            self.residual_multiplier = lambda x: x

    def call(self, inputs, training=False):
        nodes, edges, receivers, senders, global_latent, node_graph_idx, edge_graph_idx = inputs
        batch_size = global_latent.shape[0]
        # ---------------EDGE step---------------
        # nodes mapped to edges
        received_attributes = self.gather(nodes, receivers)
        sent_attributes = self.gather(nodes, senders)
        global_latent_edge = self.gather(global_latent, edge_graph_idx)

        # global_latent affects each node the same
        edges_update = tf.concat([edges, tf.squeeze(sent_attributes), tf.squeeze(received_attributes),
                                  global_latent_edge], axis=1)
        edges_update = self.edge_model(edges_update)

        # ---------------NODE step---------------
        adjacent_edges = tf.math.unsorted_segment_sum(edges_update, receivers, batch_size * FLAGS.n_nodes)
        node_shaped_global = self.gather(global_latent, node_graph_idx)
        nodes_update = tf.concat([nodes, adjacent_edges, node_shaped_global], axis=1)
        nodes_update = self.node_model(nodes_update)

        # ---------------GLOBAL step---------------
        node_aggregate = tf.math.unsorted_segment_sum(nodes_update, node_graph_idx, num_segments=batch_size)
        edge_aggregate = tf.math.unsorted_segment_sum(edges_update, edge_graph_idx, num_segments=batch_size)
        global_latent_update = tf.concat([node_aggregate, edge_aggregate, global_latent], 1)
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
    def __init__(self, edge_model_fn, node_model_fn, nodes_dropout=0., edges_dropout=0.):
        super().__init__()
        # instantiates the MLPs here
        self.edge_model = edge_model_fn()
        self.node_model = node_model_fn()
        self.gather = xpu.gather
        self.node_dropout = xpu.Dropout(rate=nodes_dropout, noise_shape=(FLAGS.batch_size * FLAGS.n_nodes, 1))
        self.edge_dropout = xpu.Dropout(rate=edges_dropout, noise_shape=(FLAGS.batch_size * FLAGS.n_edges, 1))
        if FLAGS.rn_multiplier in ('constant', 'softplus'):
            self.residual_multiplier = ResnetMultiplier(method=FLAGS.rn_multiplier)
        else:
            self.residual_multiplier = lambda x: x

    def call(self, inputs, training=False):
        nodes, edges, receivers, senders, global_latent, node_graph_idx, edge_graph_idx = inputs
        # ---------------EDGE step---------------
        # nodes mapped to edges
        received_attributes = self.gather(nodes, receivers)
        sent_attributes = self.gather(nodes, senders)
        edges_update = tf.concat([edges, sent_attributes, received_attributes], axis=1)
        edges_update = self.edge_model(edges_update)

        # ---------------NODE step---------------
        adjacent_edges = tf.math.unsorted_segment_sum(edges_update, receivers, nodes.shape[0])
        nodes_update = tf.concat([nodes, adjacent_edges], axis=1)
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
        return nodes, edges, receivers, senders, tf.zeros([FLAGS.batch_size, 1]), node_graph_idx, edge_graph_idx


class GinDecoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        nodes, edges, receivers, senders, global_latent, node_graph_idx, edge_graph_idx = inputs
        graphwise_means = tf.math.unsorted_segment_mean(nodes, node_graph_idx, FLAGS.batch_size)
        output_prob = self.output_layer(graphwise_means)
        return output_prob


class GraphIsomorphismLayer(tf.keras.layers.Layer):
    def __init__(self, get_mlp, use_edges=True, epsilon=None, edge_dim=100, dropout=0.):
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
            self.dropout = xpu.Dropout(rate=dropout, noise_shape=(FLAGS.batch_size * FLAGS.n_nodes, 1))
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
        n_nodes = tf.keras.backend.shape(nodes)[0]

        sent_attributes = tf.gather(nodes, senders)
        message = tf.math.unsorted_segment_sum(sent_attributes, receivers, n_nodes)

        if self.use_edges:
            # todo add self-loop with special lookup value
            embedded_edges = self.edge_encoder(edges)
            sum_edge_embeddings = tf.math.unsorted_segment_sum(embedded_edges, receivers, n_nodes)
            mlp_input = ((1 + self.epsilon) * nodes + message) + sum_edge_embeddings
        else:
            mlp_input = ((1 + self.epsilon) * nodes + message)

        node_update = self.mlp(mlp_input)
        if self.dropout is not None:
            node_update = self.dropout(node_update)

        return node_update, edges, receivers, senders, global_latent, node_graph_idx, edge_graph_idx


class MLP(keras.layers.Layer):
    def __init__(self, n_layers, n_hidden, n_out, dense_kernel_regularization=0., activate_final=False, name='MLP'):
        super().__init__(name=name)
        self.mlp = tf.keras.models.Sequential()
        for _ in range(n_layers - 1):
            self.mlp.add(keras.layers.Dense(n_hidden, activation=None,
                                            kernel_regularizer=keras.regularizers.l2(dense_kernel_regularization)))
            # norm PRIOR to the activation
            if FLAGS.mlp_norm == 'layer_hidden':
                self.mlp.add(xpu.LayerNormalization())

            self.mlp.add(tf.keras.layers.Activation('relu'))

        self.mlp.add(keras.layers.Dense(n_out, activation=None,
                                        kernel_regularizer=keras.regularizers.l2(dense_kernel_regularization)))

        if FLAGS.mlp_norm == 'layer_output':
            self.mlp.add(xpu.LayerNormalization())
        if activate_final:
            self.mlp.add(tf.keras.layers.Activation('relu'))

    def call(self, inputs, training=False):
        return self.mlp(inputs)


class ResnetMultiplier(keras.layers.Layer):
    def __init__(self, method='constant', name='MLP'):
        super().__init__(name=name)
        self.method = method

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if self.method == 'constant':
            init_val = 0.
        elif self.method == 'softplus':
            init_val = -5.
        else:
            raise NotImplementedError()

        init = tf.constant_initializer(init_val)
        self.w = self.add_weight('value', shape=[1], initializer=init, dtype=dtype)
        self.built = True

    def call(self, inputs, _training=True):
        if self.method == 'constant':
            return self.w * inputs
        elif self.method == 'softplus':
            return tf.nn.softplus(self.w) * inputs
        else:
            raise NotImplementedError("Invalid method for resnet multiplier")
