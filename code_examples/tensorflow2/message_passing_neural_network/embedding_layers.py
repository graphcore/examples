# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims


class AtomEncoder(tf.keras.layers.Layer):

    def __init__(self, emb_dim):
        super().__init__()

        self.emb_dim = emb_dim
        self.atom_embedding_list = []
        for dim in get_atom_feature_dims():
            self.atom_embedding_list.append(tf.keras.layers.Embedding(dim, emb_dim))

    def call(self, inputs, training=True):
        x_embedding = tf.zeros([inputs.shape[0], self.emb_dim], dtype=inputs.dtype)
        for i in range(inputs.shape[1]):
            x_embedding += self.atom_embedding_list[i](inputs[:, i])

        return x_embedding


class BondEncoder(tf.keras.layers.Layer):

    def __init__(self, emb_dim):
        super().__init__()

        self.emb_dim = emb_dim
        self.bond_embedding_list = []
        for dim in get_bond_feature_dims():
            self.bond_embedding_list.append(tf.keras.layers.Embedding(dim, emb_dim))

    def call(self, inputs, training=True):
        x_embedding = tf.zeros([inputs.shape[0], self.emb_dim], dtype=inputs.dtype)
        for i in range(inputs.shape[1]):
            x_embedding += self.bond_embedding_list[i](inputs[:, i])

        return x_embedding
