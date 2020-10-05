# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import sys
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.initializers import glorot_normal as init_glorot
from tensorflow.python import ipu
from .transformer_baseclass import Transformer
sys.path.append("../../")  # ipu_sparse_ops
from ipu_sparse_ops.model_baseclass import SparseModel  # noqa: E402


class DynsparseTransformer(Transformer, SparseModel):
    def __init__(self, params, *args, **kwargs):
        self.encoder_k = None
        self.encoder_v = None
        # Dynsparse transformer is both a transformer and a sparse model
        Transformer.__init__(self, params, *args, **kwargs)
        SparseModel.__init__(self, params, *args, **kwargs)

    def embedding(self, x, src, debug_name=''):  # x[batch_size*sequence_length] -> x[batch_size*sequence_length, embedding_length]
        inshape = x.shape
        assert len(inshape) == 2, f"Input to embedding lookup has shape {inshape}, but should be a 2D tensor"

        vocab_len = self.source_vocab_length if src else self.target_vocab_length
        sequence_length = self.source_sequence_length if src else self.target_sequence_length
        with self.namescope(f'{debug_name}embedding'):
            with self.namescope('token_lut'):
                # Embedding gets reused later on in the prediction layer
                dict_name = "source_" if src else "target_"
                dict_name = dict_name + "embedding_dict"
                embedding_dict = tf.get_variable(dict_name, (vocab_len, self.embedding_length), self.embedding_dtype, init_glorot)
                x = ipu.embedding_ops.embedding_lookup(embedding_dict, x)
                x = self.norm(x)

            # Add the positional encodings
            x = self.position_encoder(x, sequence_length, debug_name=debug_name)

            # normalize before the projection to hidden length
            x = self.norm(x)

            # The embedding length is decoupled from the hidden length
            with self.namescope("up_project_factorized_embedding"):
                x = self.sparseLinear(x, self.sparsity, self.config.hidden_length, self.compute_dense_grad, True)
                x = ipu.nn_ops.gelu(x)

        return x, embedding_dict

    def projection(self, x):  # x[..,embedding_length] -> x[..,target_vocab_length]
        with self.namescope('projection'):
            with self.namescope('down_project_factorized_embedding'):
                # Project from hidden_length to embedding_length so we can reuse the
                # embedding look-up table
                x = self.sparseLinear(x, self.sparsity, self.config.embedding_length, self.compute_dense_grad, True)
                x = ipu.nn_ops.gelu(x)
                x = self.norm(x)

            with self.namescope('decoder_logits'):
                # The look-up table is shared with the dense embedding layer
                if self.include_embedding:
                    decoder_w = tf.transpose(self.tied_embedding, name="decoder_w")
                else:
                    decoder_w = tf.get_variable("decoder_w", (self.embedding_length, self.target_vocab_length), x.dtype, init_glorot)

                # Optionally attach a bias to each token
                if self.include_projection_bias:
                    decoder_b = np.zeros([self.target_vocab_length], dtype=x.dtype) + 1.0 / self.target_vocab_length
                    # no chance of start token (small)
                    if self.target_bos_id is not None:
                        decoder_b[self.target_bos_id] = 1.0 / (self.target_vocab_length * 2)

                    # every sequence has an end token, but most
                    # sequences are shorter than sequence length
                    if self.target_eos_id is not None:
                        decoder_b[self.target_eos_id] = 2.0 / self.target_sequence_length

                    decoder_b = np.log(decoder_b)
                    decoder_b = tf.get_variable("bias", (self.target_vocab_length), x.dtype, decoder_b.astype(x.dtype))
                    x = tf.nn.xw_plus_b(x, decoder_w, decoder_b)
                else:
                    x = tf.matmul(x, decoder_w)
        return x

    def norm(self, x):  # -> x
        with self.namescope('layernorm'):
            param_initializers = {
                "beta": tf.initializers.constant(0.0, x.dtype),
                "gamma": tf.initializers.constant(0.1, x.dtype)
            }
            x = ipu.normalization_ops.group_norm(x, groups=1, param_initializers=param_initializers)
        return x

    def feed_forward(self, x):  # -> x
        with self.namescope('ffn'):
            with self.namescope('1'):
                x = self.sparseLinear(x, self.sparsity, self.ff_length, self.compute_dense_grad, True)

            with self.namescope('activation'):
                x = ipu.nn_ops.gelu(x)
                x = self.dropout(x)

            with self.namescope('2'):
                x = self.sparseLinear(x, self.sparsity, self.hidden_length, self.compute_dense_grad, True)
        return x

    def attention(self, in_q, in_k, in_v, mask=None, debug_name=''):
        """
        [batch_size, sequence_length, hidden_length] -> [B, S, 1 ,H]
        """
        with self.namescope(debug_name):
            # Prepend (head) dimension
            in_q_r = tf.expand_dims(in_q, axis=-2)
            in_k_r = tf.expand_dims(in_k, axis=-2)
            in_v_r = tf.expand_dims(in_v, axis=-2)

            # Parameter dimensions
            b_shape = self.attention_heads * self.qkv_length

            # Queries
            use_bias = self.include_attention_biases
            with self.namescope('q'):
                batch_size, sequence_length, _ = in_q.shape.as_list()
                q = self.sparseLinear(in_q_r, self.sparsity, b_shape, self.compute_dense_grad, use_bias)
                # Extract heads and transpose [B, S, heads, qkv_len] -> [B, heads, S, qkv_len]
                q = tf.reshape(q, [self.batch_size, sequence_length, self.attention_heads, self.qkv_length])
                q = tf.transpose(q, perm=[0, 2, 1, 3])  # [B, S, heads, qkv_len] -> [B, heads, S, qkv_len]
            # Keys
            with self.namescope('k'):
                batch_size, sequence_length, _ = in_k.shape.as_list()
                k = self.sparseLinear(in_k_r, self.sparsity, b_shape, self.compute_dense_grad, use_bias)
                # Extract heads and transpose [B, S, heads, qkv_len] -> [B, heads, S, qkv_len]
                k = tf.reshape(k, [self.batch_size, sequence_length, self.attention_heads, self.qkv_length])
                k = tf.transpose(k, perm=[0, 2, 3, 1])
            # Values
            with self.namescope('v'):
                batch_size, sequence_length, _ = in_v.shape.as_list()
                v = self.sparseLinear(in_v_r, self.sparsity, b_shape, self.compute_dense_grad, use_bias)
                # Extract heads and transpose  [B, S, heads, qkv_len] -> [B, heads, S, qkv_len]
                v = tf.reshape(v, [self.batch_size, sequence_length, self.attention_heads, self.qkv_length])  # split heads
                v = tf.transpose(v, perm=[0, 2, 1, 3])

            # Dense attention calculation
            with self.namescope('interaction'):
                x = tf.matmul(q, k, name="token_to_token")
                c = tf.constant(1 / np.sqrt(self.qkv_length), x.dtype)
                x = tf.multiply(x, c)

                # "Memory mask" e.g. causal
                if mask is not None and (in_q == in_k):  # only in self attention
                    x = tf.add(x, mask, name="attention_mask")

                # Take softmax across the last axis (B, heads, S1, S2) pick (0, 1, 2, 3<-)
                x = tf.nn.softmax(x, axis=3)

            # Dense projection using the interaction matrix x
            with self.namescope('z'):
                # x[B, heads, seq_len1, seq_len2] @ v[B, heads, seq_len2, qkv_length]
                z = tf.matmul(x, v)
                # [B, heads, seq_len1, qkv_length] -> [B, seq_len1, heads, qkv_length]
                z = tf.transpose(z, perm=[0, 2, 1, 3])
                # [B, seq_len1, heads, qkv_length] -> [B, seq_len1, heads*qkv_length]
                z = tf.reshape(z, [self.batch_size, -1, self.attention_heads * self.qkv_length])

            # Project back to hidden_length using a sparse matmul
            with self.namescope('out'):
                z = self.sparseLinear(z, self.sparsity, self.hidden_length, self.compute_dense_grad, use_bias)
                # Typically no activation here

        # z[batch_size, sequence_length, hidden_length]
        return z

    def position_encoder(self, x, sequence_length, debug_name=''):  # -> x
        pe = np.zeros([sequence_length, self.embedding_length], np.float32)

        # pe = sin(pos/(10000**(2i/d_model))) or cos(pos/10000**((2i+1)/d_model))
        # pos = sequence dimension index. i embedding dimension index.
        d_model = self.embedding_length
        pos = np.arange(0, sequence_length).reshape(-1, 1)
        i = np.arange(0, d_model, 2).reshape(1, -1)
        # Calculate div_term in log space.
        div_term = np.exp(-(np.log(10000.0) * i / d_model))  # TODO: Check this against paper

        # Even dims sin, odd dims cos.
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)

        # make sure pe is copied correctly to device
        pe = np.ascontiguousarray(pe.astype(np.float32))

        with self.namescope('positional_embeddings'):
            pe = tf.constant(pe, dtype=x.dtype)
            pe = self.norm(pe)
            x = self.add(x, pe)
            x = self.dropout(x)
        return x

    def add(self, a, b):
        return tf.add(a, b)

    def dropout(self, x):  # -> x
        if self.dropout_keep_prob is not None:
            x = tf.nn.dropout(x, self.dropout_keep_prob)
        return x

    def namescope(self, debug_string):
        return tf.variable_scope(debug_string, use_resource=True, reuse=tf.AUTO_REUSE)
