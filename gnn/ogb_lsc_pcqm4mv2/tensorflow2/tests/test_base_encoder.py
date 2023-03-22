# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import tensorflow as tf
import pytest
from tensorflow.python import ipu
import tensorflow as tf
from tensorflow.keras import mixed_precision
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from model.hybrid.layers import get_default_mlp

from model.encoders import base_encoder


class TestBaseEncoder:
    @classmethod
    def setup_class(cls):
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)

        config = ipu.config.IPUConfig()
        config.auto_select_ipus = 1
        config.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
        ipu.utils.configure_ipu_system(config)
        cls.inputs = {}
        cls.encoder = None

    def create_encoder(
        self,
        n_embedding_channels=56,
        encoder_dropout_rate=0.1,
        node_latent=128,
        edge_latent=32,
        encoder_latent=32,
        global_latent=32,
    ):
        # Features
        shape = (16, 41, 9)
        self.shape = shape
        self.inputs = {
            "node_feat": tf.keras.Input(
                shape=shape, dtype=tf.int32, tensor=tf.constant(np.ones(shape), dtype=tf.int32)
            ),
            "edge_feat": tf.keras.Input(
                shape=shape, dtype=tf.int32, tensor=tf.constant(np.ones(shape), dtype=tf.int32)
            ),
            "lap_eig_vecs": tf.keras.Input(
                shape=shape, dtype=tf.int32, tensor=tf.constant(np.ones(shape), dtype=tf.int32)
            ),
            "lap_eig_vals": tf.keras.Input(
                shape=(16, 41, 9, 1), dtype=tf.int32, tensor=tf.constant(np.ones((16, 41, 9, 1)), dtype=tf.int32)
            ),
            "random_walk_landing_probs": tf.keras.Input(
                shape=shape, dtype=tf.int32, tensor=tf.constant(np.ones(shape), dtype=tf.int32)
            ),
            "receivers": tf.keras.Input(
                shape=shape, dtype=tf.int32, tensor=tf.constant(np.ones(shape), dtype=tf.int32)
            ),
            "senders": tf.keras.Input(shape=shape, dtype=tf.int32, tensor=tf.constant(np.ones(shape), dtype=tf.int32)),
            "node_graph_idx": tf.keras.Input(
                shape=shape, dtype=tf.int32, tensor=tf.constant(np.ones(shape), dtype=tf.int32)
            ),
            "edge_graph_idx": tf.keras.Input(
                shape=shape, dtype=tf.int32, tensor=tf.constant(np.ones(shape), dtype=tf.int32)
            ),
            "shortest_path_distances": tf.keras.Input(
                shape=shape, dtype=tf.int32, tensor=tf.constant(np.ones((shape[0], shape[1], shape[1])), dtype=tf.int32)
            ),
        }

        def f_x(latent):
            from model.gnn.layers import Dense

            return Dense(latent)

        get_default_node_mlp = lambda node_latent: f_x(node_latent)
        get_default_edge_mlp = lambda edge_latent: f_x(edge_latent)
        get_default_encoder_mlp = lambda encoder_latent: f_x(encoder_latent)

        self.encoder = base_encoder.BaseEncoder(
            self.inputs,
            node_model_fn=lambda: get_default_node_mlp(node_latent=node_latent),
            edge_model_fn=lambda: get_default_edge_mlp(edge_latent=edge_latent),
            encoder_model_fn=lambda: get_default_encoder_mlp(encoder_latent=encoder_latent),
            use_globals=True,
            use_edges=True,
            n_attn_heads=8,
            global_latent=global_latent,
            n_embedding_channels=n_embedding_channels,
            encoder_dropout_rate=encoder_dropout_rate,
            micro_batch_size=16,
            n_nodes_per_pack=41,
            n_graphs_per_pack=1,
            node_feature_dims=get_atom_feature_dims(),
            edge_feature_dims=get_bond_feature_dims(),
            max_shortest_path_distance=200,
        )
        assert self.encoder.atom_encoder
        assert self.encoder.bond_encoder
        assert self.encoder.lap_eig_val_encodings
        assert self.encoder.lap_eig_vec_encodings
        assert self.encoder.random_walk_encodings

    @pytest.mark.parametrize(
        "n_embedding_channels, encoder_dropout_rate, node_latent, edge_latent, encoder_latent, global_latent",
        [
            (56, 0.1, 128, 128, 128, 128),
            (9, 0.1, 64, 256, 16, 8),
        ],
    )
    def test_basic_encoder(
        self, n_embedding_channels, encoder_dropout_rate, node_latent, edge_latent, encoder_latent, global_latent
    ):
        self.create_encoder(
            n_embedding_channels=n_embedding_channels,
            encoder_dropout_rate=encoder_dropout_rate,
            node_latent=node_latent,
            edge_latent=edge_latent,
            encoder_latent=encoder_latent,
            global_latent=global_latent,
        )
        encode_tests = self.encoder(self.inputs)
        assert encode_tests["node_feat"].shape == (self.shape[0], self.shape[1], node_latent)
        assert encode_tests["edge_feat"].shape == (self.shape[0], self.shape[1], edge_latent)
        assert encode_tests["lap_eig_vecs"].shape == (self.shape[0], self.shape[1], encoder_latent)
        assert encode_tests["lap_eig_vals"].shape == (self.shape[0], self.shape[1], encoder_latent)
        assert encode_tests["random_walk_landing_probs"].shape == (self.shape[0], self.shape[1], encoder_latent)
        assert encode_tests["receivers"].shape == self.shape
        assert encode_tests["senders"].shape == self.shape
        assert encode_tests["node_graph_idx"].shape == self.shape
        assert encode_tests["edge_graph_idx"].shape == self.shape
        assert encode_tests["shortest_path_distances"].shape == (16, 41, 41, 8)
