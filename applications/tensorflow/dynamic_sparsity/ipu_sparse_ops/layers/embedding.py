# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
Sparse embedding layers.
"""
import os
import tensorflow.compat.v1 as tf
from ipu_sparse_ops import sparse, layers
from logging import getLogger


logger = getLogger(os.path.basename(__file__))


class SparseTiedEmbedding:
    def __init__(self, name: str, projection: layers.SparseFcLayer, embedding_grad_scale: float = 1.0):
        """
        Construct a new 'SparseTiedEmbedding' object. For use in (for example) language
        models where the input and output embedding weights are shared. It is recommended
        to create these layers using the factory functions e.g: 'from_sparse_projection'.

        :param name: Name string for the layer (used as a variable scope).
        :param projection: A sparse fully connected layer (see SparseFcLayer).
                           The rows of the sparse matrix will be the embedding
                           vectors.
        :param embedding_grad_scale: A scale factor applied to only the embedding's
                                     contribution to the weight gradient. Allows for
                                     fine tuning the update to the shared weights.
        """
        self.name = name
        self.projection = projection
        self.embedding_grad_scale = embedding_grad_scale
        logger.info(f"Creating embedding layer '{self.name}': tied to "
                    f"projection layer: '{self.projection.name}'")

    @classmethod
    def from_sparse_projection(cls, name: str, projection: layers.SparseFcLayer):
        """
        Factory function to create a 'SparseTiedEmbedding' layer from an existing
        SparseFCLayer (which provides the embedding weights and computes the tied
        projection).

        :param name: Name string for the layer (used as a variable scope).
        :param projection: A sparse fully connected layer (see SparseFcLayer).
                              The rows of the sparse matrix will be the embedding
                              vectors.
        """
        return cls(name, projection)

    def __call__(self, ids):
        """
        Build and return the op to execute the embedding layer. The ids will
        be used to index the rows of the tied embedding/projection weights.

        This also builds the projection layer which can be called later: self.projection(input)
        :param ids: Indices used for the embedding lookup.
        """
        # Make sure we build the projection layer first as we reuse its weights
        with tf.variable_scope(self.projection.name, reuse=tf.AUTO_REUSE, use_resource=True):
            self.projection.build()

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE, use_resource=True):
            # Flatten the input because the sparse embedding only work on a 1D vector
            hidden_length = self.projection.weights.spec.input_size
            input_shape = ids.shape.as_list()
            ids = tf.reshape(ids, [-1])

            # Cast the tokens to fp32 because that is what poplar expects
            ids = tf.cast(ids, tf.float32)

            x = sparse.embedding_with_vars(
                self.projection.weights.spec,
                ids,
                self.projection.weights.matmul_options,
                self.projection.get_values_var(),
                self.projection.get_metainfo_var(),
                self.embedding_grad_scale)

            # Reshape to the expected shape from an embedding, e.g. [batch-size, seq_len, hidden_lengh]
            return tf.reshape(x, input_shape + [hidden_length])
