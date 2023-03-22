# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import xpu
from model.gnn.embedding_layers import AtomEncoder, BondEncoder
from model.gnn.layers import MLP, Dense
from tensorflow.python import ipu


class BaseEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        inputs,
        node_model_fn,
        edge_model_fn,
        encoder_model_fn,
        use_globals=False,
        global_latent=None,
        encoder_latent=None,
        use_edges=False,
        n_embedding_channels=None,
        n_attn_heads=None,
        encoder_dropout_rate=0.0,
        override_encoder_dropout={},
        one_hot_embeddings=False,
        micro_batch_size=None,
        eigv_rand_sign_flip=False,
        n_nodes_per_pack=None,
        n_edges_per_pack=None,
        n_graphs_per_pack=None,
        node_feature_dims=None,
        edge_feature_dims=None,
        max_shortest_path_distance=None,
        num_gaussian_kernels=None,
        gaussian_kernel_init=None,
        gaussian_kernel_scale=None,
        gaussian_kernel_epsilon=None,
        use_distance_sum_feature=None,
        masked_features=[],
    ):
        super().__init__()
        self.use_globals = use_globals
        self.use_edges = use_edges
        self.micro_batch_size = micro_batch_size
        self.n_graphs_per_pack = n_graphs_per_pack
        self.encoder_dropout_rate = encoder_dropout_rate
        self.eigv_rand_sign_flip = eigv_rand_sign_flip
        self.max_shortest_path_distance = max_shortest_path_distance
        self.masked_features = masked_features
        self.num_gaussian_kernels = num_gaussian_kernels
        self.gaussian_kernel_init = gaussian_kernel_init
        self.gaussian_kernel_scale = gaussian_kernel_scale
        self.gaussian_kernel_epsilon = gaussian_kernel_epsilon
        self.use_distance_sum_feature = use_distance_sum_feature
        categorical_feats = []

        allowed_features = ["node_feat", "edge_feat", "lap_eig_vecs", "lap_eig_vals", "random_walk_landing_probs"]
        incorrect_keys = set(override_encoder_dropout.keys()) - set(allowed_features)
        assert len(incorrect_keys) == 0, f"{incorrect_keys} in override_encoder_dropout not allowed"

        for feature_name, feature in inputs.items():
            # Encoders for all optional features.
            # These may be concatenated together into node / edge features
            # They may also be wanted to be used independatly
            #     - i.e. Using Laplacian vectors as positional encodings for transformer
            if feature_name == "node_feat":
                self.atom_encoder = AtomEncoder(n_embedding_channels, one_hot_embeddings, node_feature_dims)
                self.node_model = node_model_fn()
                node_dropout_rate = override_encoder_dropout.get("node_feat", encoder_dropout_rate)
                if node_dropout_rate > 0:
                    self.node_dropout = xpu.Dropout(
                        rate=node_dropout_rate, noise_shape=(micro_batch_size, n_nodes_per_pack, 1)
                    )
            elif feature_name == "edge_feat":
                if self.use_edges:
                    self.bond_encoder = BondEncoder(n_embedding_channels, one_hot_embeddings, edge_feature_dims)
                    self.edge_model = edge_model_fn()
                    edge_dropout_rate = override_encoder_dropout.get("edge_feat", encoder_dropout_rate)
                    if edge_dropout_rate > 0:
                        self.edge_dropout = xpu.Dropout(
                            rate=edge_dropout_rate, noise_shape=(micro_batch_size, n_edges_per_pack, 1)
                        )
            elif feature_name == "lap_eig_vecs":
                self.lap_eig_vec_encodings = encoder_model_fn()
                lap_eig_vec_dropout_rate = override_encoder_dropout.get("lap_eig_vecs", encoder_dropout_rate)
                if lap_eig_vec_dropout_rate > 0:
                    self.lap_eig_vec_dropout = xpu.Dropout(
                        rate=lap_eig_vec_dropout_rate, noise_shape=(micro_batch_size, n_nodes_per_pack, 1)
                    )
            elif feature_name == "lap_eig_vals":
                self.lap_eig_val_encodings = encoder_model_fn()
                lap_eig_val_dropout_rate = override_encoder_dropout.get("lap_eig_vals", encoder_dropout_rate)
                if lap_eig_val_dropout_rate > 0:
                    self.lap_eig_val_dropout = xpu.Dropout(
                        rate=lap_eig_val_dropout_rate, noise_shape=(micro_batch_size, n_nodes_per_pack, 1)
                    )
            elif feature_name == "random_walk_landing_probs":
                self.random_walk_encodings = encoder_model_fn()
                random_walk_dropout_rate = override_encoder_dropout.get(
                    "random_walk_landing_probs", encoder_dropout_rate
                )
                if random_walk_dropout_rate > 0:
                    self.random_walk_dropout = xpu.Dropout(
                        rate=random_walk_dropout_rate, noise_shape=(micro_batch_size, n_nodes_per_pack, 1)
                    )
            elif feature_name == "ogb_bond_lengths":
                if self.use_edges:
                    if self.num_gaussian_kernels > 0:
                        self.ogb_bond_lengths_encoder = GaussianKernelEncoder(
                            self.num_gaussian_kernels,
                            self.gaussian_kernel_init,
                            self.gaussian_kernel_scale,
                            self.gaussian_kernel_epsilon,
                        )
                    self.ogb_bond_lengths_encodings = encoder_model_fn()
                    ogb_bond_lengths_dropout_rate = override_encoder_dropout.get(
                        "ogb_bond_lengths", encoder_dropout_rate
                    )
                    if ogb_bond_lengths_dropout_rate > 0:
                        self.ogb_bond_lengths_dropout = xpu.Dropout(
                            rate=ogb_bond_lengths_dropout_rate, noise_shape=(micro_batch_size, n_edges_per_pack, 1)
                        )
            elif feature_name == "relative_features":
                self.relative_features_encodings = encoder_model_fn()
                relative_features_dropout_rate = override_encoder_dropout.get("relative_features", encoder_dropout_rate)
                if relative_features_dropout_rate > 0:
                    self.relative_features_dropout = xpu.Dropout(
                        rate=relative_features_dropout_rate, noise_shape=(micro_batch_size, n_edges_per_pack, 1)
                    )
            elif feature_name == "shortest_path_distances":
                if max_shortest_path_distance:
                    self.shortest_path_distances_encoder = ShortestPathDistanceEncoder(
                        max_shortest_path_distance=max_shortest_path_distance, n_attn_heads=n_attn_heads
                    )
            elif feature_name == "atom_distances":
                self.atom_distances_encoder = GaussianKernelEncoder(
                    self.num_gaussian_kernels,
                    self.gaussian_kernel_init,
                    self.gaussian_kernel_scale,
                    self.gaussian_kernel_epsilon,
                )
                self.atom_distances_gaussian_mlp = MLP(
                    n_layers=2, n_hidden=num_gaussian_kernels, n_out=n_attn_heads, activation_function="gelu"
                )
                if self.use_distance_sum_feature:
                    self.atom_distances_sum_mlp = Dense(encoder_latent, use_bias=False)

            elif feature_name == "centrality_encoding":
                self.centrality_encoder = tf.keras.layers.Embedding(
                    n_nodes_per_pack + 1,  # +1 makes space for index 0 = padding
                    n_embedding_channels,
                    mask_zero=True,
                    name="CentralityEmbedding",
                )

            elif feature_name in [
                "receivers",
                "senders",
                "node_graph_idx",
                "edge_graph_idx",
                "node_mask",
                "edge_mask",
                "direction_vector",
                "nan_in_conformer",
            ]:
                pass
            else:
                raise NotImplementedError(f"No Encoder Implemented for {feature_name}.")

        # Add the global encoder
        if self.use_globals:
            self.global_latent = tf.keras.layers.Embedding(1, int(global_latent))

    def call(self, inputs=None, training=False, *args, **kwargs):
        activation_dtype = tf.float16 if "float16" in self.dtype_policy.name else tf.float32
        # - Combine these features into these objects here
        # - i.e. all_nodes_feats += node_feat_encodings, += lap_eig_vec_encodings
        all_feature_encodings = {}
        node_mask = inputs.get("node_mask", None)
        edge_mask = inputs.get("edge_mask", None)
        zero_value = -128  # special value marked in packed_batch_generator

        if "shortest_path_distances" in inputs.keys():
            attn_mask, attn_bias = get_attn_masks(inputs["shortest_path_distances"], activation_dtype)
            all_feature_encodings["attn_mask"], all_feature_encodings["attn_bias"] = attn_mask, attn_bias

        for feature_name, feature in inputs.items():
            # Encoders for all optional features.
            # These may be concatenated together into node / edge features
            # They may also be wanted to be used independatly
            #     - i.e. Using Laplacian vectors as positional encodings for transformer

            if feature_name in self.masked_features:
                # get zero_mask for features
                N_dims = 3 if feature_name in ("shortest_path_distances", "atom_distances") else 2
                reduce_dims = range(len(feature.get_shape().as_list()))[N_dims:]
                zero_mask = tf.where(
                    tf.reduce_all(feature == zero_value, axis=reduce_dims),
                    tf.cast(0, feature.dtype),
                    tf.cast(1, feature.dtype),
                )
                # may possible want to do a pre encoder mask to stop Nans being generated
                # feature *= zero_mask

            if feature_name == "node_feat":
                nodes = self.atom_encoder(feature)
                node_feats_encodings = self.node_model(nodes, mask=node_mask)
                all_feature_encodings[feature_name] = node_feats_encodings
            elif feature_name == "edge_feat":
                if self.use_edges:
                    edge_update = self.bond_encoder(feature)
                    edge_update = self.edge_model(edge_update, mask=edge_mask)
                    all_feature_encodings[feature_name] = edge_update
            elif feature_name == "lap_eig_vecs":
                lap_eig_vec_encodings = self.lap_eig_vec_encodings(
                    feature, feature_name=feature_name, eigv_rand_sign_flip=self.eigv_rand_sign_flip, mask=node_mask
                )
                if hasattr(self, "lap_eig_vec_dropout"):
                    lap_eig_vec_encodings = self.lap_eig_vec_dropout(lap_eig_vec_encodings)
                all_feature_encodings[feature_name] = lap_eig_vec_encodings
            elif feature_name == "lap_eig_vals":
                feature = tf.squeeze(feature)
                lap_eig_val_encodings = self.lap_eig_val_encodings(feature, mask=node_mask)
                if hasattr(self, "lap_eig_val_dropout"):
                    lap_eig_val_encodings = self.lap_eig_val_dropout(lap_eig_val_encodings)
                all_feature_encodings[feature_name] = lap_eig_val_encodings
            elif feature_name == "random_walk_landing_probs":
                random_walk_encodings = self.random_walk_encodings(feature, mask=node_mask)
                if hasattr(self, "random_walk_dropout"):
                    random_walk_encodings = self.random_walk_dropout(random_walk_encodings)
                all_feature_encodings[feature_name] = random_walk_encodings
            elif feature_name == "ogb_bond_lengths":
                if self.use_edges:
                    feature = tf.expand_dims(feature, -1)
                    if self.num_gaussian_kernels > 0:
                        feature = self.ogb_bond_lengths_encoder(feature)
                    ogb_bond_lengths_encodings = self.ogb_bond_lengths_encodings(feature)
                    if hasattr(self, "ogb_bond_lengths_dropout"):
                        ogb_bond_lengths_encodings = self.ogb_bond_lengths_dropout(ogb_bond_lengths_encodings)
                    all_feature_encodings[feature_name] = ogb_bond_lengths_encodings
            elif feature_name == "relative_features":
                relative_features_encodings = self.relative_features_encodings(feature, mask=edge_mask)
                if hasattr(self, "relative_features_dropout"):
                    relative_features_encodings = self.relative_features_dropout(relative_features_encodings)
                all_feature_encodings[feature_name] = relative_features_encodings
            elif feature_name == "shortest_path_distances":
                if hasattr(self, "shortest_path_distances_encoder"):
                    all_feature_encodings[feature_name] = (
                        self.shortest_path_distances_encoder(feature) * all_feature_encodings["attn_mask"]
                    )
            elif feature_name == "centrality_encoding":
                all_feature_encodings[feature_name] = self.centrality_encoder(feature)
            elif feature_name == "atom_distances":
                feature = tf.expand_dims(feature, -1)
                atom_distance_feats = self.atom_distances_encoder(feature) * all_feature_encodings["attn_mask"]
                all_feature_encodings[feature_name] = (
                    self.atom_distances_gaussian_mlp(atom_distance_feats) * all_feature_encodings["attn_mask"]
                )
                if self.use_distance_sum_feature:
                    distance_sum_feats = tf.reduce_sum(atom_distance_feats, axis=-2, keepdims=False)
                    distance_sum_feats = self.atom_distances_sum_mlp(distance_sum_feats)
                    # special case because distance_sum_feats is fully dependent on atom_distance_feats
                    if feature_name in self.masked_features:
                        sum_feats_zero_mask = zero_mask
                        sum_feats_zero_mask = tf.reduce_max(sum_feats_zero_mask, axis=-2, keepdims=False)
                        while len(sum_feats_zero_mask.get_shape().as_list()) < len(
                            distance_sum_feats.get_shape().as_list()
                        ):
                            sum_feats_zero_mask = tf.expand_dims(sum_feats_zero_mask, -1)
                        distance_sum_feats *= sum_feats_zero_mask

                    all_feature_encodings["distance_sum_feats"] = distance_sum_feats
            elif feature_name in ["receivers", "senders", "node_graph_idx", "edge_graph_idx", "direction_vector"]:
                all_feature_encodings[feature_name] = feature
            elif feature_name in ["node_mask", "edge_mask", "nan_in_conformer"]:
                pass
            else:
                raise NotImplementedError(f"No Encoder Implemented for {feature_name}.")

            if feature_name in self.masked_features:
                while len(zero_mask.get_shape().as_list()) < len(
                    all_feature_encodings[feature_name].get_shape().as_list()
                ):
                    zero_mask = tf.expand_dims(zero_mask, -1)

                all_feature_encodings[feature_name] *= tf.cast(
                    zero_mask, dtype=all_feature_encodings[feature_name].dtype
                )

        if self.use_globals:
            # each graph has the same learned global embedding
            global_latent = self.global_latent(
                tf.constant([0] * self.micro_batch_size * self.n_graphs_per_pack, dtype=tf.int32)
            )
            global_latent = tf.reshape(global_latent, [self.micro_batch_size, self.n_graphs_per_pack, -1])
            all_feature_encodings["global_latent"] = global_latent

        return all_feature_encodings


class ConcatFeatures(tf.keras.layers.Layer):
    def __init__(self, node_latent=304, edge_latent=32, use_edges=True, concat_mode="concat_all", input_names=[]):
        super().__init__()
        # Lists of node and edge features to concat appropriately.
        # Default empty list means nothing to concat
        possible_node_features = [
            "lap_eig_vals",
            "lap_eig_vecs",
            "random_walk_landing_probs",
            "centrality_encoding",
            "distance_sum_feats",
        ]

        possible_edge_features = ["relative_features", "rdkit_bond_lengths", "ogb_bond_lengths"]
        possible_attention_biases = ["attn_bias", "shortest_path_distances", "atom_distances"]

        base_node_feats = ["node_feat"]
        base_edge_feats = ["edge_feat"]
        extra_node_feats = list(sorted(set(input_names).intersection(possible_node_features)))
        extra_edge_feats = list(sorted(set(input_names).intersection(possible_edge_features)))

        if concat_mode == "concat_all":
            self.node_feats_to_concat = base_node_feats + extra_node_feats
            self.edge_feats_to_concat = base_edge_feats + extra_edge_feats
            self.node_feats_to_sum = []
            self.edge_feats_to_sum = []
        elif concat_mode == "sum_all":
            self.node_feats_to_concat = []
            self.edge_feats_to_concat = []
            self.node_feats_to_sum = base_node_feats + extra_node_feats
            self.edge_feats_to_sum = base_edge_feats + extra_edge_feats
        elif concat_mode == "sum_extras":
            self.node_feats_to_concat = base_node_feats
            self.edge_feats_to_concat = base_edge_feats
            self.node_feats_to_sum = extra_node_feats
            self.edge_feats_to_sum = extra_edge_feats
        else:
            raise ValueError(f"concat_mode {concat_mode} not recognised")

        self.attention_biases_to_sum = sorted(set(input_names).intersection(possible_attention_biases))

        self.use_edges = use_edges
        self.use_attn_bias = len(self.attention_biases_to_sum) > 0
        self.node_mlp = tf.keras.layers.Dense(node_latent)
        if self.use_edges:
            self.edge_mlp = tf.keras.layers.Dense(edge_latent)

    def call(self, inputs, *args, **kwargs):
        # Node Features - start with node features and concatenate

        node_feat_concat = []
        node_feat_sum = []
        edge_feat_concat = []
        edge_feat_sum = []

        for feature_name in sorted(list(inputs.keys())):
            if feature_name in self.node_feats_to_concat:
                node_feat_concat += [inputs[feature_name]]
            if feature_name in self.node_feats_to_sum:
                node_feat_sum += [inputs[feature_name]]
            if self.use_edges:
                if feature_name in self.edge_feats_to_concat:
                    edge_feat_concat += [inputs[feature_name]]
                if feature_name in self.edge_feats_to_sum:
                    edge_feat_sum += [inputs[feature_name]]

        if node_feat_sum != []:
            node_feat_concat += [tf.math.add_n(node_feat_sum)]
        if edge_feat_sum != []:
            edge_feat_concat += [tf.math.add_n(edge_feat_sum)]

        # Final MLP layers to reshape the node and edge to correct latent sizes
        node_feat = tf.keras.layers.Concatenate(axis=-1)(node_feat_concat)
        inputs["node_feat"] = self.node_mlp(node_feat)
        if self.use_edges:
            edge_feat = tf.keras.layers.Concatenate(axis=-1)(edge_feat_concat)
            inputs["edge_feat"] = self.edge_mlp(edge_feat)
        else:
            inputs["edge_feat"] = tf.constant(0)

        if self.use_attn_bias:
            assert (
                "attn_mask" in inputs.keys()
            ), "Packing mask is integrated into shortest_path_distances, so it must be present for MHSA"
            summands = [inputs[name] for name in self.attention_biases_to_sum if name in inputs.keys()]
            attn_bias = tf.keras.layers.Add()(summands) if len(summands) > 1 else summands[0]
            # (batch, nodes, nodes, heads) -> (batch, heads, nodes, nodes)
            attn_bias = tf.transpose(attn_bias, (0, 3, 1, 2))
            inputs["attn_bias"] = attn_bias

        # Ensure outputs are ordered, and add dummy values so pipelining doesn't break
        outputs = [
            inputs.get(name, tf.constant(0))
            for name in [
                "node_feat",
                "edge_feat",
                "senders",
                "receivers",
                "global_latent",
                "node_graph_idx",
                "edge_graph_idx",
                "attn_bias",
                "direction_vector",
            ]
        ]
        return outputs


class ShortestPathDistanceEncoder(tf.keras.layers.Layer):
    def __init__(self, max_shortest_path_distance, n_attn_heads):
        super().__init__()
        self.max_shortest_path_distance = max_shortest_path_distance
        self.embedding = tf.keras.layers.Embedding(
            max_shortest_path_distance + 1, n_attn_heads, mask_zero=True, name="ShortestPathDistanceEmbedding"
        )

    def call(self, raw_inputs, training=True):
        # raw_inputs is (n_nodes, n_nodes), where -1 implies no attention, 0 implies untrainable
        # attention (i.e. padding), and other values mean logical distances offset by 1
        distances = tf.clip_by_value(raw_inputs, 0, self.max_shortest_path_distance + 1)
        feats = self.embedding(distances, training=training)
        zero = tf.constant(0, dtype=feats.dtype)
        feats = tf.where(raw_inputs[..., tf.newaxis] != -1, feats, zero)
        return feats


def get_attn_masks(shortest_path_feats, activation_dtype):
    # Mask attention between nodes that aren't connected
    minus_inf_val = (
        float("-inf") if activation_dtype == tf.float32 else -10000
    )  # very small value that rounds to 0 after exp
    minus_inf = tf.constant(minus_inf_val, dtype=activation_dtype)
    one = tf.constant(1, dtype=activation_dtype)
    zero = tf.constant(0, dtype=activation_dtype)
    attn_mask = tf.where(shortest_path_feats[..., tf.newaxis] != -1, one, zero)
    attn_bias = tf.where(shortest_path_feats[..., tf.newaxis] != -1, zero, minus_inf)
    return attn_mask, attn_bias


class GaussianKernelEncoder(tf.keras.layers.Layer):
    def __init__(self, num_gaussian_kernels, init=(0, 1.5), scale=1.0, epsilon=1e-5):
        super().__init__()
        self.num_gaussian_kernels = num_gaussian_kernels
        assert num_gaussian_kernels > 0
        self.init_low = init[0]
        self.init_high = init[1]
        self.scale = scale
        self.epsilon = epsilon
        # Leaving this part here in case we need in the future
        # self.lambda_ij = tf.keras.layers.Embedding(num_edge_types, 1, embeddings_initializer='ones')
        # self.beta_ij = tf.keras.layers.Embedding(num_edge_types, 1, embeddings_initializer='zeros')

    def build(self, input_shape):
        # initialisation values are chosen with the assumption of mean_only [x/2*mean(x)]

        K = self.num_gaussian_kernels
        self.gaussian_mean = self.add_weight(
            "gaussian_mean",
            shape=[self.num_gaussian_kernels],
            initializer=tf.random_uniform_initializer(minval=self.init_low, maxval=self.init_high),
            trainable=True,
        )
        self.gaussian_std = self.add_weight(
            "gaussian_std",
            shape=[self.num_gaussian_kernels],
            initializer=tf.random_uniform_initializer(minval=self.init_low, maxval=self.init_high),
            trainable=True,
        )

    def call(self, inputs, activation_dtype=None, training=True):
        input_dtype = inputs.dtype
        # Leaving this part here in case we need in the future
        # lambda_ij = self.lambda_ij(edge_types)  # lambda
        # beta_ij = self.beta_ij(edge_types)  # beta
        # distances = lambda_ij * distances + beta_ij

        # scale to make sure max kernel values are roughly around 1
        mean = self.gaussian_mean
        std = tf.math.abs(self.gaussian_std) + self.epsilon

        while len(mean.get_shape().as_list()) < len(inputs.get_shape().as_list()):
            mean = tf.expand_dims(mean, 0)
            std = tf.expand_dims(std, 0)

        pi = 3.141592657
        pre_exp_factor = (2 * pi) ** 0.5
        inputs, mean, std = (tf.cast(x, dtype=tf.float32) for x in [inputs, mean, std])
        output = self.scale * tf.math.exp(-0.5 * (((inputs - mean) / std) ** 2)) / (pre_exp_factor * std)
        return tf.cast(output, dtype=input_dtype)
