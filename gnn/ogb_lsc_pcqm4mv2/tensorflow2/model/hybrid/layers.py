# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import math

import tensorflow as tf

import xpu
from model.gnn.embedding_layers import AtomEncoder, BondEncoder
from model.gnn.layers import MLP, InteractionNetworkLayer, GraphDropout
from model.gnn.aggregators import gather


def get_default_mlp(
    activate_final=None,
    activation_function=None,
    name=None,
    n_mlp_layers=None,
    exp_ratio=None,
    n_latent=None,
    expand_first_hidden=True,
    weight_dtype=None,
    mlp_norm=None,
    prenorm=None,
    preact=None,
):
    n_hidden = int(n_latent * exp_ratio)
    n_hidden_first = n_hidden if expand_first_hidden else n_latent
    return MLP(
        n_layers=n_mlp_layers,
        n_hidden_first=n_hidden_first,
        n_hidden=n_hidden,
        n_out=n_latent,
        activate_final=activate_final,
        activation_function=activation_function,
        mlp_norm=mlp_norm,
        weight_dtype=weight_dtype,
        name=name,
        prenorm=prenorm,
        preact=preact,
    )


def build_gnn_mlps(
    activation_function,
    expand_first_hidden,
    mlp_norm,
    node_latent,
    node_exp_ratio,
    node_mlp_layers,
    node_prenorm,
    edge_latent,
    edge_exp_ratio,
    edge_mlp_layers,
    edge_prenorm,
    global_latent,
    global_exp_ratio,
    global_mlp_layers,
    global_prenorm,
    encoder_latent,
    encoder_exp_ratio,
    encoder_mlp_layers,
    encoder_prenorm,
    encoder_act_fn,
    decoder_mlp_layers,
    decoder_hidden,
    decoder_prenorm,
    weight_dtype,
):
    def node_mlp(activate_final, name):
        return get_default_mlp(
            activate_final=activate_final,
            activation_function=activation_function,
            name=name,
            n_latent=node_latent,
            exp_ratio=node_exp_ratio,
            n_mlp_layers=node_mlp_layers,
            expand_first_hidden=expand_first_hidden,
            weight_dtype=weight_dtype,
            mlp_norm=mlp_norm,
            prenorm=node_prenorm,
        )

    def edge_mlp(activate_final, name):
        return get_default_mlp(
            activate_final=activate_final,
            activation_function=activation_function,
            name=name,
            n_latent=edge_latent,
            exp_ratio=edge_exp_ratio,
            n_mlp_layers=edge_mlp_layers,
            expand_first_hidden=expand_first_hidden,
            weight_dtype=weight_dtype,
            mlp_norm=mlp_norm,
            prenorm=edge_prenorm,
        )

    def global_mlp(activate_final, name):
        return get_default_mlp(
            activate_final=activate_final,
            activation_function=activation_function,
            name=name,
            n_latent=global_latent,
            exp_ratio=global_exp_ratio,
            n_mlp_layers=global_mlp_layers,
            expand_first_hidden=expand_first_hidden,
            weight_dtype=weight_dtype,
            mlp_norm=mlp_norm,
            prenorm=global_prenorm,
        )

    def encoder_mlp(activate_final, name):
        return get_default_mlp(
            activate_final=activate_final,
            activation_function=encoder_act_fn or activation_function,
            name=name,
            n_latent=encoder_latent,
            exp_ratio=encoder_exp_ratio,
            n_mlp_layers=encoder_mlp_layers,
            expand_first_hidden=expand_first_hidden,
            weight_dtype=weight_dtype,
            mlp_norm=mlp_norm,
            prenorm=encoder_prenorm,
            preact=False,
        )

    def decoder_mlp(n_out, name="output_logits"):
        return MLP(
            n_layers=decoder_mlp_layers,
            n_hidden=decoder_hidden,
            n_out=n_out,
            activate_final=False,
            activation_function=activation_function,
            mlp_norm=mlp_norm,
            weight_dtype=weight_dtype,
            prenorm=decoder_prenorm,
            name=name,
        )

    return node_mlp, edge_mlp, global_mlp, encoder_mlp, decoder_mlp


class GPSLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        node_model,
        edge_model,
        global_model,
        micro_batch_size,
        nodes_per_pack,
        edges_per_pack,
        graphs_per_pack,
        node_latent,
        ffn_dim,
        n_attn_heads,
        attention_dropout_rate,
        ffn_dropout_rate,
        gnn_output_dropout_rate,
        mhsa_output_dropout_rate,
        ffn_output_dropout_rate,
        gnn_node_dropout,
        gnn_edge_dropout,
        gnn_global_dropout,
        gnn_edge_dropout_loc,
        use_edges,
        use_globals,
        gather_from,
        scatter_to,
        aggregators,
        concat_globals_to,
        direct_neighbour_aggregation,
        node_combine_method,
        rn_multiplier,
        gather_scatter,
        layer_spec,
        skip_global_out,
        graph_dropout_rate,
        override_graph_dropout_rate,
        layer_depth_frac,
        hybrid_mpnn_extra_node_residual,
    ):

        super().__init__()
        self.node_latent = node_latent
        self.ffn_dim = ffn_dim
        self.attention_dropout_rate = attention_dropout_rate
        self.hybrid_mpnn_extra_node_residual = hybrid_mpnn_extra_node_residual
        self.ffn_dropout_rate = ffn_dropout_rate
        self.n_attn_heads = n_attn_heads
        self.spec = layer_spec
        self.use_globals = use_globals

        mpnn_graph_dropout = (
            graph_dropout_rate if override_graph_dropout_rate.MPNN is None else override_graph_dropout_rate.MPNN
        )
        ffn_graph_dropout = (
            graph_dropout_rate if override_graph_dropout_rate.FFN is None else override_graph_dropout_rate.FFN
        )
        mhsa_graph_dropout = (
            graph_dropout_rate if override_graph_dropout_rate.MHSA is None else override_graph_dropout_rate.MHSA
        )
        mpnn_graph_dropout *= layer_depth_frac
        ffn_graph_dropout *= layer_depth_frac
        mhsa_graph_dropout *= layer_depth_frac

        if "MPNN" in self.spec:
            self.gnn_module = InteractionNetworkLayer(
                node_model_fn=node_model,
                use_edges=use_edges,
                edge_model_fn=edge_model,
                use_globals=use_globals,
                global_model_fn=global_model,
                node_dropout=gnn_node_dropout,
                edge_dropout=gnn_edge_dropout,
                global_dropout=gnn_global_dropout,
                edge_dropout_loc=gnn_edge_dropout_loc,
                gather_from=gather_from,
                scatter_to=scatter_to,
                aggregators=aggregators,
                concat_globals_to=concat_globals_to,
                direct_neighbour_aggregation=direct_neighbour_aggregation,
                node_combine_method=node_combine_method,
                micro_batch_size=micro_batch_size,
                n_nodes_per_pack=nodes_per_pack,
                n_edges_per_pack=edges_per_pack,
                n_graphs_per_pack=graphs_per_pack,
                rn_multiplier=rn_multiplier,
                output_scale=1.0,
                gather_scatter=gather_scatter,
                graph_dropout_rate=mpnn_graph_dropout,
                skip_global_out=skip_global_out,
            )
            self.gnn_norm = xpu.LayerNormalization(axis=-1)
            if gnn_output_dropout_rate > 0.0:
                self.gnn_output_dropout = xpu.Dropout(gnn_output_dropout_rate)

        if "MHSA" in self.spec:
            self.transformer_module = BiasedSelfAttentionLayer(
                node_latent, attention_dropout_rate, n_attn_heads, prediction_head=False
            )
            self.transformer_norm = xpu.LayerNormalization(axis=-1)
            if mhsa_output_dropout_rate > 0.0:
                self.mhsa_output_dropout = xpu.Dropout(mhsa_output_dropout_rate)
            self.mhsa_graph_dropout = GraphDropout(mhsa_graph_dropout, graphs_per_pack, gather_scatter)

        if "FFN" in self.spec:
            self.FFN = FFNLayer(node_latent, ffn_dim, ffn_dropout_rate)
            self.ffn_norm = xpu.LayerNormalization(axis=-1)
            if ffn_output_dropout_rate > 0.0:
                self.ffn_output_dropout = xpu.Dropout(ffn_output_dropout_rate)
            self.ffn_graph_dropout = GraphDropout(ffn_graph_dropout, graphs_per_pack, gather_scatter)

    def call(self, inputs, training=True):
        (
            nodes,
            edges,
            senders,
            receivers,
            globals_,
            node_graph_idxs,
            egde_graph_idxs,
            attn_bias,
            direction_vector,
            *_,
        ) = inputs
        nodes_updates = []
        globals_updates = []
        # LOCAL MODULE, e.g. GNN #
        if "MPNN" in self.spec:
            nodes_update, edges, _, _, globals_update, _, _, _ = self.gnn_module(inputs)
            if hasattr(self, "gnn_output_dropout"):
                nodes_update = self.gnn_output_dropout(nodes_update, training=training)
            # No GraphDropout here, that's already done inside the MPNN layer
            if self.hybrid_mpnn_extra_node_residual:
                nodes_update = nodes_update + nodes
            nodes_update = self.gnn_norm(nodes_update, training=training)
            nodes_updates.append(nodes_update)
            globals_updates.append(globals_update)

        # GLOBAL MODULE, e.g. multi-headed self-attention
        if "MHSA" in self.spec:
            nodes_update = self.transformer_module(nodes, attn_bias, training=training)
            if "float16" in self.dtype_policy.name:
                nodes_update = tf.cast(nodes_update, tf.float16)
            if hasattr(self, "mhsa_output_dropout"):
                nodes_update = self.mhsa_output_dropout(nodes_update, training=training)
            nodes_update = self.mhsa_graph_dropout(nodes_update, node_graph_idxs, training=training)
            nodes_update = nodes_update + nodes  # Residual
            nodes_update = self.transformer_norm(nodes_update, training=training)
            nodes_updates.append(nodes_update)

        # Recombine
        assert nodes_updates, f'Layer "{self.spec}" doesn\'t modify nodes'
        nodes = nodes_updates.pop()
        for update in nodes_updates:
            nodes = nodes + update
        if self.use_globals and globals_updates:
            globals_ = globals_updates.pop()
            for update in globals_updates:
                globals_ = globals_ + update

        # FFN to mix global and local
        if "FFN" in self.spec:
            ffn_nodes = self.FFN(nodes, training=training)
            if hasattr(self, "ffn_output_dropout"):
                ffn_nodes = self.ffn_output_dropout(ffn_nodes, training=training)
            ffn_nodes = self.ffn_graph_dropout(ffn_nodes, node_graph_idxs, training=training)
            ffn_nodes = ffn_nodes + nodes  # Residual
            nodes = self.ffn_norm(ffn_nodes, training=training)

        return (
            nodes,
            edges,
            senders,
            receivers,
            globals_,
            node_graph_idxs,
            egde_graph_idxs,
            attn_bias,
            direction_vector,
        )


class BiasedSelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, attention_dropout_rate, n_attn_heads, prediction_head=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dropout_rate = attention_dropout_rate
        self.n_attn_heads = n_attn_heads
        self.prediction_head = prediction_head

        self.head_size = hidden_dim // n_attn_heads
        self.scale = 1.0 / math.sqrt(self.head_size)

        init_rad = hidden_dim**-0.5
        uniform_initialiser = tf.keras.initializers.RandomUniform(-init_rad, init_rad)
        if self.prediction_head:
            self.q, self.k, self.v = (
                tf.keras.layers.Dense(
                    hidden_dim,
                    kernel_initializer=uniform_initialiser,
                    bias_initializer=uniform_initialiser,
                    kernel_regularizer=None,
                    use_bias=True,
                    activation=None,
                )
                for _ in range(3)
            )
        else:
            self.q, self.k, self.v, self.projection = (
                tf.keras.layers.Dense(
                    hidden_dim,
                    kernel_initializer=uniform_initialiser,
                    bias_initializer=uniform_initialiser,
                    kernel_regularizer=None,
                    use_bias=True,
                    activation=None,
                )
                for _ in range(4)
            )

        self.attention_dropout = xpu.Dropout(attention_dropout_rate)

    def call(self, x, attn_bias, training=True):
        batch, nodes, hidden = x.shape
        q = tf.reshape(self.q(x), (batch, nodes, self.n_attn_heads, self.head_size))
        k = tf.reshape(self.k(x), (batch, nodes, self.n_attn_heads, self.head_size))
        v = tf.reshape(self.v(x), (batch, nodes, self.n_attn_heads, self.head_size))

        q = tf.transpose(q, (0, 2, 1, 3))  # [batch, heads, nodes, head_size]
        k = tf.transpose(k, (0, 2, 3, 1))  # [batch, heads, head_size, nodes]
        v = tf.transpose(v, (0, 2, 1, 3))  # [batch, heads, nodes, head_size]

        attn = tf.matmul(q, k)  # [batch, heads, nodes, nodes]
        attn = attn * tf.cast(self.scale, dtype=attn.dtype)
        attn = attn + tf.cast(attn_bias, dtype=attn.dtype)  # attn_bias [batch, heads, nodes, nodes]
        attn = tf.nn.softmax(logits=attn, axis=-1)  # [batch, heads, nodes, nodes]
        attn = self.attention_dropout(attn, training=training)  # [batch, heads, nodes, nodes]
        attn_prob = attn
        if self.prediction_head:
            return attn_prob, v
        feats = tf.matmul(attn, v)  # [batch, heads, nodes, head_size]
        feats = tf.transpose(feats, (0, 2, 1, 3))
        feats = tf.reshape(feats, (batch, nodes, hidden))
        # feats [batch, nodes, hidden]
        feats = self.projection(feats, training=training)
        return feats


class FFNLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, ffn_dim, activation_dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.activation_dropout_rate = activation_dropout_rate

        r = hidden_dim**-0.5
        uniform_initialiser = tf.keras.initializers.RandomUniform(-r, r)

        self.dense1 = tf.keras.layers.Dense(
            ffn_dim,
            activation=xpu.gelu,
            kernel_initializer=uniform_initialiser,
            bias_initializer=uniform_initialiser,
            use_bias=True,
        )
        self.dense2 = tf.keras.layers.Dense(
            hidden_dim, kernel_initializer=uniform_initialiser, bias_initializer=uniform_initialiser, use_bias=True
        )
        if activation_dropout_rate:
            self.activation_dropout = xpu.Dropout(activation_dropout_rate)

    def call(self, x, training=True):
        x = self.dense1(x, training=training)
        if hasattr(self, "activation_dropout"):
            x = self.activation_dropout(x, training=training)
        x = self.dense2(x, training=training)
        return x
