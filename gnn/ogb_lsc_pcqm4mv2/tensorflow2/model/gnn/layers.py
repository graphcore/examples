# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import logging

import tensorflow as tf
from keras.layers import Activation
from tensorflow import keras
from tensorflow.python import ipu

import xpu
from model.gnn.aggregators import GenericAggregator, gather

# ======== General MPNN functions ========


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        decoder_mode=None,
        n_graphs_per_pack=None,
        gather_scatter=None,
        decoder_model_fn=None,
        noisy_node_model=None,
        use_edges=False,
        use_globals=False,
        pooling_aggregators=["sum"],
        noisy_nodes=False,
        noisy_edges=False,
        node_feature_sum=None,
        edge_feature_sum=None,
    ):
        super().__init__()
        # instantiates the MLPs here
        self.decoder_model = decoder_model_fn(1, name="main_decoder")
        self.decoder_mode = decoder_mode
        self.n_graphs_per_pack = n_graphs_per_pack
        self.pooling_aggregator = GenericAggregator(
            aggregators=pooling_aggregators, gather_scatter_method=gather_scatter
        )
        self.use_edges = use_edges
        self.use_globals = use_globals
        self.noisy_nodes = noisy_nodes
        self.noisy_edges = noisy_edges

        if noisy_nodes:
            self.node_aux_layer = noisy_node_model(node_feature_sum, name="NNode_aux")
        if noisy_edges:
            self.edge_aux_layer = noisy_node_model(edge_feature_sum, name="NEdge_aux")

    def call(self, inputs, training=False):
        nodes, edges, _, _, global_latent, node_graph_idx, *_ = inputs

        if self.decoder_mode == "node_global":
            node_inputs_to_global_decoder = self.pooling_aggregator(nodes, node_graph_idx, self.n_graphs_per_pack)
            inputs_to_decoder = tf.concat([node_inputs_to_global_decoder, global_latent], axis=-1)
        elif self.decoder_mode == "node":
            inputs_to_decoder = self.pooling_aggregator(nodes, node_graph_idx, self.n_graphs_per_pack)
        elif self.decoder_mode == "global":
            inputs_to_decoder = global_latent
        else:
            raise ValueError("Pick a relevant decoder mode")
        logits = self.decoder_model(inputs_to_decoder)
        output = [logits]
        if self.noisy_nodes:
            output += [self.node_aux_layer(nodes)]
        if self.noisy_edges:
            output += [self.edge_aux_layer(edges)]

        return output


class InteractionNetworkLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        node_model_fn=None,
        use_edges=True,
        edge_model_fn=None,
        use_globals=False,
        global_model_fn=None,
        node_dropout=0.0,
        edge_dropout=0.0,
        global_dropout=0.0,
        edge_dropout_loc="before_residual_add",
        edge_dna_dropout=0.0,
        gather_from="both",
        scatter_to="receivers",
        concat_globals_to=["nodes"],  # edges, nodes, both
        direct_neighbour_aggregation=False,
        node_combine_method="concat",
        aggregators=["sum"],
        output_scale=1.0,
        cumulative_scale=1.0,
        micro_batch_size=None,
        n_nodes_per_pack=None,
        n_edges_per_pack=None,
        n_graphs_per_pack=None,
        rn_multiplier=None,
        gather_scatter=None,
        graph_dropout_rate=0.0,
        skip_global_out=False,
    ):
        super().__init__()
        # instantiates the MLPs here
        self.node_model = node_model_fn()
        self.node_dropout = xpu.Dropout(rate=node_dropout, noise_shape=(micro_batch_size, n_nodes_per_pack, 1))
        self.use_edges = use_edges
        self.edge_dropout_loc = edge_dropout_loc
        if use_edges:
            self.edge_model = edge_model_fn()
            if edge_dropout > 0:
                self.edge_dropout = xpu.Dropout(
                    rate=edge_dropout, seed=generate_dropout_seed(), noise_shape=(micro_batch_size, n_edges_per_pack, 1)
                )
            else:
                self.edge_dropout = lambda x, training: x
            if edge_dna_dropout > 0:
                self.edge_dna_dropout = {
                    "senders": xpu.Dropout(rate=edge_dropout, noise_shape=(micro_batch_size, n_edges_per_pack, 1)),
                    "receivers": xpu.Dropout(rate=edge_dropout, noise_shape=(micro_batch_size, n_edges_per_pack, 1)),
                }
            else:
                self.edge_dna_dropout = {}
                self.edge_dna_dropout["senders"] = lambda x, training: x
                self.edge_dna_dropout["receivers"] = lambda x, training: x

        self.use_globals = use_globals
        if use_globals and not skip_global_out:
            self.global_model_fn = global_model_fn()
            self.global_dropout = xpu.Dropout(rate=global_dropout, noise_shape=(micro_batch_size, n_graphs_per_pack, 1))

        self.gather_from = gather_from
        self.scatter_to = scatter_to
        self.concat_globals_to = concat_globals_to
        self.direct_neighbour_aggregation = direct_neighbour_aggregation
        self.node_combine_method = node_combine_method
        self.output_scale = tf.constant(output_scale, tf.float32)
        self.cumulative_scale = tf.constant(cumulative_scale, tf.float32)
        if rn_multiplier in ("constant", "softplus"):
            self.residual_multiplier = ResnetMultiplier(method=rn_multiplier)
        else:
            self.residual_multiplier = lambda x: x

        self.n_nodes_per_pack = n_nodes_per_pack
        self.n_edges_per_pack = n_edges_per_pack
        self.n_graphs_per_pack = n_graphs_per_pack
        self.gather_scatter = gather_scatter
        self.skip_global_out = skip_global_out

        self.aggregator = GenericAggregator(aggregators=aggregators, gather_scatter_method=gather_scatter)
        self.pooling_aggregator = GenericAggregator(aggregators=aggregators, gather_scatter_method=gather_scatter)
        self.graph_dropout = GraphDropout(graph_dropout_rate, n_graphs_per_pack, gather_scatter)

    def residual_add(self, identity_path, compute_path):
        compute_path = self.residual_multiplier(compute_path)
        # scale compute path so that it matches the identity scale
        compute_path *= tf.cast(self.cumulative_scale, compute_path.dtype)
        out = identity_path + compute_path
        # scale be some constant factor to stop growth of activation scale
        out *= tf.cast(self.output_scale, out.dtype)
        return out

    def gather_features(self, input_features, senders, receivers):
        out = []

        receiver_features = gather(input_features, receivers, gather_scatter_method=self.gather_scatter)
        sender_features = gather(input_features, senders, gather_scatter_method=self.gather_scatter)

        if self.gather_from == "receivers":
            out.append(receiver_features)

        if self.gather_from == "senders":
            out.append(sender_features)

        if self.gather_from == "both":
            if self.node_combine_method == "sum":
                out.append(receiver_features + sender_features)
            elif self.node_combine_method == "concat":
                out.append(receiver_features)
                out.append(sender_features)
            else:
                raise ValueError(f"node_combine_method {self.node_combine_method} not recognised.")

        return out, sender_features, receiver_features

    def aggregate_features(self, input_features, senders, receivers, sender_features, receiver_features, dim):
        out = []
        aggregated_features = []

        if self.scatter_to in ["receivers", "both"]:
            if self.direct_neighbour_aggregation:
                message = tf.concat([input_features, sender_features], axis=-1)
            else:
                message = input_features
            aggregated_features.append(self.aggregator(message, receivers, dim))

        if self.scatter_to in ["senders", "both"]:
            if self.direct_neighbour_aggregation:
                message = tf.concat([input_features, receiver_features], axis=-1)
            else:
                message = input_features
            aggregated_features.append(self.aggregator(message, senders, dim))

        if self.node_combine_method == "sum" and self.scatter_to == "both":
            out.append(aggregated_features[0] + aggregated_features[1])
        else:
            out.extend(aggregated_features)

        return out

    def get_global_features(self, global_latent, idx, case):
        if self.use_globals:
            accepted_cases = ["nodes", "edges"]
            if case not in accepted_cases:
                raise ValueError(f"{case} not in {accepted_cases}")
            incorrect_vals = set(self.concat_globals_to) - set(accepted_cases)
            if len(incorrect_vals) == 0:
                if case in self.concat_globals_to:
                    return gather(global_latent, idx, gather_scatter_method=self.gather_scatter)
                else:
                    return None
            else:
                raise ValueError(f"{incorrect_vals} not a valid entry for concat_to_globals")
        else:
            return None

    def call(self, inputs, training=False, *args, **kwargs):
        (
            nodes_input,
            edges_input,
            senders,
            receivers,
            global_latent_input,
            node_graph_idx,
            edge_graph_idx,
            attn_mask,
            *_,
        ) = inputs

        # ---------------EDGE step---------------
        # nodes mapped to edges
        edge_model_input, sender_nodes, receiver_nodes = self.gather_features(nodes_input, senders, receivers)

        if self.use_edges:
            edge_model_input.append(edges_input)
            edge_model_input = tf.concat(edge_model_input, axis=-1)

            global_latent_edge = self.get_global_features(global_latent_input, edge_graph_idx, "edges")
            if global_latent_edge is not None:
                edge_model_input = tf.concat([edge_model_input, global_latent_edge], axis=-1)
            edges = self.edge_model(edge_model_input)
            if "before_scatter" in self.edge_dropout_loc:
                edges = self.edge_dropout(edges, training=training)
        else:
            edges = tf.concat(edge_model_input, axis=-1)

        # ---------------NODE step---------------

        if self.use_edges:
            sender_nodes = self.edge_dna_dropout["senders"](sender_nodes, training=training)
            receiver_nodes = self.edge_dna_dropout["receivers"](receiver_nodes, training=training)
        node_model_input = self.aggregate_features(
            edges, senders, receivers, sender_nodes, receiver_nodes, self.n_nodes_per_pack
        )
        node_model_input.append(nodes_input)
        global_latent_node = self.get_global_features(global_latent_input, node_graph_idx, "nodes")
        if global_latent_node is not None:
            node_model_input.append(global_latent_node)
        nodes = tf.concat(node_model_input, axis=-1)
        nodes = self.node_model(nodes)

        # ---------------GLOBAL step---------------
        if self.use_globals and not self.skip_global_out:
            global_latent = [global_latent_input]
            global_latent.append(self.pooling_aggregator(nodes, node_graph_idx, self.n_graphs_per_pack))
            if self.use_edges:
                global_latent.append(self.pooling_aggregator(edges, edge_graph_idx, self.n_graphs_per_pack))
            global_latent = tf.concat(global_latent, -1)
            global_latent = self.global_model_fn(global_latent)
        else:
            global_latent = global_latent_input

        # ---------------- Stochastic depth  ---------------
        nodes = self.graph_dropout(nodes, node_graph_idx)
        if self.use_edges:
            edges = self.graph_dropout(edges, edge_graph_idx)
        if self.use_globals:
            global_latent = self.graph_dropout(global_latent, None)

        # dropout before the residual block`
        nodes = self.node_dropout(nodes, training=training)
        nodes = self.residual_add(nodes_input, nodes)
        if self.output_scale != 1.0:
            nodes *= tf.cast(self.output_scale, nodes.dtype)

        if self.use_edges:
            if self.edge_dropout_loc == "before_residual_add":
                edges = self.edge_dropout(edges, training=training)
            edges = self.residual_add(edges_input, edges)
            if self.output_scale != 1.0:
                edges *= tf.cast(self.output_scale, edges.dtype)

        if self.use_globals and not self.skip_global_out:
            global_latent = self.global_dropout(global_latent, training=training)
            global_latent = self.residual_add(global_latent_input, global_latent)
            if self.output_scale != 1.0:
                global_latent *= tf.cast(self.output_scale, global_latent.dtype)

        return nodes, edges, senders, receivers, global_latent, node_graph_idx, edge_graph_idx, attn_mask


# ======== General functions ========


class LayerNorm(xpu.LayerNormalization):
    def call(self, inputs, mask=None, training=None):
        return super().call(inputs, training=True)


class MLP(keras.layers.Layer):
    def __init__(
        self,
        n_layers=None,
        n_hidden=None,
        n_hidden_first=None,
        n_out=None,
        activate_final=False,
        activation_function="relu",
        name="MLP",
        weight_dtype=None,
        mlp_norm=None,
        prenorm=None,
        preact=None,
    ):
        super().__init__(name=name)
        if not n_hidden_first:
            n_hidden_first = n_hidden

        self.n_layers = n_layers
        self.activate_final = activate_final
        self.mlp_norm = mlp_norm
        self.Dense_layers = []
        self.Norm_layers = []
        self.preact = preact

        self.prenorm_layer = LayerNorm() if prenorm else None

        for i in range(n_layers - 1):
            self.Dense_layers += [
                Dense(n_hidden_first if i == 0 else n_hidden, weight_dtype=weight_dtype, name=f"{name}_{i+1}")
            ]
            if mlp_norm == "layer_hidden" or mlp_norm == "all_layers":
                self.Norm_layers += [LayerNorm()]

        self.Dense_layers += [Dense(n_out, weight_dtype=weight_dtype, name=f"{name}_{n_layers}")]

        if mlp_norm == "layer_output" or mlp_norm == "all_layers":
            self.Norm_layers += [LayerNorm()]

        self.act_fn = Activation(_apply_activation_function(activation_function))

    def call(self, x, training=False, mask=None, *args, **kwargs):
        if "feature_name" in kwargs and kwargs["feature_name"] == "lap_eig_vecs":
            # randomize the signs of eigen vectors
            eigv_rand_sign_flip = kwargs["eigv_rand_sign_flip"]
            if training and eigv_rand_sign_flip:
                # random initialize a uniform distribution between 0 and 1
                sign_flip = np.random.uniform(size=[x.shape[2]], low=0, high=1)
                sign_flip[sign_flip >= 0.5] = 1.0
                sign_flip[sign_flip < 0.5] = -1.0
                # apply sign_flip on the eigen vectors
                x = x * tf.cast(tf.expand_dims(sign_flip, axis=0), dtype=x.dtype)

        norm_kwargs = {"mask": mask} if mask is not None else {}
        if self.prenorm_layer is not None:
            x = self.prenorm_layer(x, training=training, **norm_kwargs)
            if self.preact:
                x = self.act_fn(x)

        for i in range(self.n_layers - 1):
            x = self.Dense_layers[i](x)
            if self.mlp_norm == "layer_hidden" or self.mlp_norm == "all_layers":
                x = self.Norm_layers[i](x, training=training, **norm_kwargs)
            x = self.act_fn(x)

        x = self.Dense_layers[-1](x)

        if self.mlp_norm == "layer_output" or self.mlp_norm == "all_layers":
            x = self.Norm_layers[-1](x, training=training, **norm_kwargs)

        if self.activate_final:
            x = self.act_fn(x)

        return x


class Dense(tf.keras.layers.Layer):
    def __init__(
        self,
        num_channels,
        use_bias=True,
        weight_dtype=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        name="dense",
    ):
        super(Dense, self).__init__(name=name)
        self.num_channels = num_channels
        self.use_bias = use_bias
        self.weight_dtype = weight_dtype
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.W = self.add_weight(
            "W",
            shape=[input_shape[-1], self.num_channels],
            initializer=self.kernel_initializer,
            dtype=self.weight_dtype,
            trainable=True,
        )
        if self.use_bias:
            self.b = self.add_weight(
                "bias",
                shape=[self.num_channels],
                initializer=self.bias_initializer,
                dtype=self.weight_dtype,
                trainable=True,
            )

    def call(self, x, *args, **kwargs):
        x = tf.matmul(x, tf.cast(self.W, dtype=x.dtype))
        if self.use_bias:
            x = tf.nn.bias_add(x, tf.cast(self.b, dtype=x.dtype), name=self.name)
        return x


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


def _apply_activation_function(activation_function):
    if activation_function == "relu":
        act_fn = tf.keras.layers.Activation("relu")
    elif activation_function == "gelu":
        act_fn = ipu.nn_ops.gelu
    elif activation_function == "swish":
        act_fn = ipu.nn_ops.swish
    else:
        logging.error(f"Activation function {activation_function} provided is not recognized")
        raise NotImplementedError
    return act_fn


def generate_dropout_seed():
    return tf.random.uniform([2], minval=0, maxval=tf.int32.max, dtype=tf.int32)


class GraphDropout(tf.keras.layers.Layer):
    def __init__(self, dropout_rate, graphs_per_pack, gather_scatter_method):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.graphs_per_pack = graphs_per_pack
        self.gather_scatter_method = gather_scatter_method

    def build(self, input_shape):
        # Mask persists between calls
        if not hasattr(self, "graph_mask"):
            dtype = tf.float16 if "float16" in tf.keras.mixed_precision.global_policy().name else tf.float32
            noise = tf.random.uniform((input_shape[0], self.graphs_per_pack, 1), dtype=dtype)
            self.graph_mask = tf.math.ceil(noise - self.dropout_rate)

    def call(self, x, graph_idxs, training=True):
        if self.dropout_rate == 0.0 or not training:
            return x

        if graph_idxs is not None:
            x_mask = gather(self.graph_mask, graph_idxs, gather_scatter_method=self.gather_scatter_method)
        else:
            # Special case, graph mask is already the right mask for x (i.e. globals_latent)
            x_mask = self.graph_mask

        keep_rate = tf.cast(1.0 - self.dropout_rate, dtype=x.dtype)
        return tf.math.divide(x, keep_rate) * x_mask
