# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow.keras import backend
import logging


class OneHotEmbedding(tf.keras.layers.Embedding):
    def call(self, inputs):
        dtype = backend.dtype(inputs)
        if dtype != "int32" and dtype != "int64":
            inputs = tf.cast(inputs, "int32")

        one_hot = tf.one_hot(inputs, self.input_dim, dtype=self.compute_dtype)
        out = one_hot @ self.embeddings
        if self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype:
            # Instead of casting the variable as in most layers, cast the output, as
            # this is mathematically equivalent but is faster.
            out = tf.cast(out, self._dtype_policy.compute_dtype)
        return out


class MultiFeatureEncoder(tf.keras.layers.Layer):
    def __init__(self, emb_dim, n_feature_dims, one_hot_embedding, name=""):
        """
        For something like an atom, which has several categorical features,
          we build a learnable embedding by summing a looked-up embedding from each
          of these several features.

        :param emb_dim: number of output dimensions from the embedding
        :param n_feature_dims: list of feature dimensions
        :param name: name for the keras layer
        """
        super().__init__(name=name)

        self.emb_dim = emb_dim
        embedding_fn = OneHotEmbedding if one_hot_embedding else tf.keras.layers.Embedding
        # one embedding table for each of the categorical feature dimensions
        self.embeddings = [embedding_fn(n, emb_dim) for n in n_feature_dims]
        logging.info(f"Items in node/edge feature: {n_feature_dims}")

    def call(self, inputs, training=True):
        output = tf.zeros([*inputs.shape[:-1], self.emb_dim], dtype=self.compute_dtype)
        for i, embedding in enumerate(self.embeddings):
            output += embedding(inputs[..., i])

        return output


class AtomEncoder(MultiFeatureEncoder):
    def __init__(self, emb_dim, one_hot_embedding, atom_feature_dims, name="AtomEncoder"):
        super().__init__(
            emb_dim=emb_dim, n_feature_dims=atom_feature_dims, one_hot_embedding=one_hot_embedding, name=name
        )


class BondEncoder(MultiFeatureEncoder):
    def __init__(self, emb_dim, one_hot_embedding, bond_feature_dims, name="BondEncoder"):
        super().__init__(
            emb_dim=emb_dim, n_feature_dims=bond_feature_dims, one_hot_embedding=one_hot_embedding, name=name
        )
