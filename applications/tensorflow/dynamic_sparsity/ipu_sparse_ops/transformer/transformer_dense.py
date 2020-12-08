# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.initializers import glorot_normal as init_glorot
from tensorflow.compat.v1.initializers import zeros as init_zeros
from tensorflow.compat.v1.initializers import variance_scaling
from tensorflow.python import ipu
from .transformer_baseclass import Transformer

weights_initializer = init_glorot
embedding_initializer = variance_scaling(mode="fan_out")


class DenseTransformer(Transformer):
    def __init__(self, opts, *args, **kwargs):
        self.encoder_k = None
        self.encoder_v = None
        super(DenseTransformer, self).__init__(opts, *args, **kwargs)

    def embedding(self, x, src, *args, **kwargs):  # x[batch_size*sequence_length] -> x[batch_size*sequence_length, embedding_length]
        inshape = x.shape
        assert len(inshape) == 2, f"Input to embedding lookup has shape {inshape}, but should be a 2D tensor"

        vocab_len = self.source_vocab_length if src else self.target_vocab_length
        assert vocab_len is not None, "Embedding vocab length must be defined"
        sequence_length = self.source_sequence_length if src else self.target_sequence_length
        with self.namescope(f'embedding'):
            with self.namescope('token_lut'):
                # Embedding gets reused later on in the prediction layer
                dict_name = "source_" if src else "target_"
                dict_name = dict_name + "embedding_dict"
                embedding_dict = tf.get_variable(dict_name, (vocab_len, self.embedding_length), self.dtype, embedding_initializer)
                x = ipu.embedding_ops.embedding_lookup(embedding_dict, x)
                x = self.norm(x)

            # Add the positional encodings
            x = self.position_encoder(x, sequence_length)

            # normalize before the projection to hidden length
            x = self.norm(x)

            # The embedding length is decoupled from the hidden length
            up_project_w = tf.get_variable("up_project_w", (self.embedding_length, self.hidden_length), x.dtype, weights_initializer)
            up_project_b = tf.get_variable("up_project_b", (self.hidden_length), x.dtype, init_zeros)
            x = tf.nn.xw_plus_b(x, up_project_w, up_project_b)
            # no non-linearity here according to ALBERT

        return x, embedding_dict

    def projection(self, x, *args, **kwargs):  # x[..,embedding_length] -> x[..,target_vocab_length]
        with self.namescope('projection'):
            with self.namescope('down_project'):
                # Project from hidden_length to embedding_length so we can reuse the
                # embedding look-up table
                down_project_w = tf.get_variable("weight", (self.hidden_length, self.embedding_length), x.dtype, weights_initializer)
                down_project_b = tf.get_variable("bias", (self.embedding_length), x.dtype, init_zeros)
                x = tf.nn.xw_plus_b(x, down_project_w, down_project_b)
                x = ipu.nn_ops.gelu(x)
                x = self.norm(x)

            with self.namescope('decoder_logits'):
                # The look-up table is shared with the embedding layer
                if not self.exclude_embedding:
                    decoder_w = tf.transpose(self.tied_embedding, name="decoder_w")
                else:
                    decoder_w = tf.get_variable("decoder_w", (self.embedding_length, self.target_vocab_length), x.dtype, weights_initializer)

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

    def feed_forward(self, x, *args, **kwargs):  # -> x
        with self.namescope('ffn'):
            with self.namescope('1'):
                w1 = tf.get_variable("weight", (self.hidden_length, self.ff_length), x.dtype, weights_initializer)
                b1 = tf.get_variable("bias", (self.ff_length), x.dtype, init_zeros)
                x = tf.nn.xw_plus_b(x, w1, b1)

            with self.namescope('activation'):
                x = ipu.nn_ops.gelu(x)
                x = self.dropout(x)

            with self.namescope('2'):
                w2 = tf.get_variable("weight", (self.ff_length, self.hidden_length), x.dtype, weights_initializer)
                b2 = tf.get_variable("bias", (self.hidden_length), x.dtype, init_zeros)
                x = tf.nn.xw_plus_b(x, w2, b2)
        return x

    def linear(self, x, w_shape, b_shape, use_bias=True):
        """
        A helper function for constructing linear layers inside
        the attention op.
        """
        w = tf.get_variable("weight", w_shape, x.dtype, weights_initializer)
        if use_bias:
            b = tf.get_variable("bias", (b_shape), x.dtype, init_zeros)
            x = tf.nn.xw_plus_b(x, w, b)
        else:
            x = tf.matmul(x, w)
        return x

    def attention(self, in_q, in_k, in_v, mask=None, is_self_attention=False, *args, **kwargs):
        """
        [batch_size, sequence_length, hidden_length] -> [B, S, 1 ,H]
        """
        # Parameter dimensions
        w_shape = [self.hidden_length, self.attention_heads * self.qkv_length]
        b_shape = [self.attention_heads * self.qkv_length]

        use_bias = not self.exclude_attention_biases

        if self.disable_concat_qkv or not is_self_attention:
            # Prepend (head) dimension
            in_q_r = tf.expand_dims(in_q, axis=-2)
            in_k_r = tf.expand_dims(in_k, axis=-2)
            in_v_r = tf.expand_dims(in_v, axis=-2)
            # Queries
            with self.namescope('q'):
                q = self.linear(in_q_r, w_shape, b_shape, use_bias)
            # Keys
            with self.namescope('k'):
                k = self.linear(in_k_r, w_shape, b_shape, use_bias)
            # Values
            with self.namescope('v'):
                v = self.linear(in_v_r, w_shape, b_shape, use_bias)
        else:
            with self.namescope('qkv'):
                # Prepend (head) dimension
                in_qkv_r = tf.expand_dims(in_q, axis=-2)
                w_shape[1] *= 3
                b_shape[0] *= 3
                qkv = self.linear(in_qkv_r, w_shape, b_shape, use_bias)

                # Extract q, k and v
                q, k, v = tf.split(qkv, 3, axis=-1)

        # Extract heads and transpose  [B, S, heads, qkv_len] -> [B, heads, S, qkv_len]
        batch_size, sequence_length, _ = in_q.shape.as_list()
        with self.namescope('q'):
            q = tf.reshape(q, [batch_size, sequence_length, self.attention_heads, self.qkv_length])
            q = tf.transpose(q, perm=[0, 2, 1, 3])  # [B, S, heads, qkv_len] -> [B, heads, S, qkv_len]
        with self.namescope('k'):
            k = tf.reshape(k, [batch_size, sequence_length, self.attention_heads, self.qkv_length])
            kt = tf.transpose(k, perm=[0, 2, 3, 1])  # [B, S, heads, qkv_len] -> [B, heads, qkv_len, S]
        with self.namescope('v'):
            v = tf.reshape(v, [batch_size, sequence_length, self.attention_heads, self.qkv_length])
            v = tf.transpose(v, perm=[0, 2, 1, 3])  # [B, S, heads, qkv_len] -> [B, heads, S, qkv_len]

        # Dense attention calculation
        with self.namescope('interaction'):
            x = tf.matmul(q, kt, name="token_to_token")
            c = tf.constant(1 / np.sqrt(self.qkv_length), x.dtype)
            x = tf.multiply(x, c)

            # "Memory mask" e.g. causal
            if mask is not None and (in_q == in_k):  # only in self attention
                x = tf.add(x, mask, name="attention_mask")

            # Take softmax across the last axis (B, heads, S1, S2) pick (0, 1, 2, 3<-)
            x = tf.nn.softmax(x, axis=3)

        with self.namescope('z'):
            # x[B, heads, seq_len1, seq_len2] @ v[B, heads, seq_len2, qkv_length]
            z = tf.matmul(x, v)

            # [B, heads, seq_len1, qkv_length] -> [B, seq_len1, heads, qkv_length]
            z = tf.transpose(z, perm=[0, 2, 1, 3])
            # [B, seq_len1, heads, qkv_length] -> [B, seq_len1, heads*qkv_length]
            z = tf.reshape(z, [batch_size, -1, self.attention_heads * self.qkv_length])

        # Project back to hidden_length
        with self.namescope('out'):
            w_shape = (self.attention_heads * self.qkv_length, self.hidden_length)
            b_shape = (self.hidden_length)
            z = self.linear(z, w_shape, b_shape, use_bias)
            # Typically no activation here

        # z[batch_size, sequence_length, hidden_length]
        return z

    def position_encoder(self, x, sequence_length):  # -> x
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

        # Add batch dimension
        pe = pe.reshape(1, *pe.shape)

        # make sure pe is copied correctly to device
        pe = np.ascontiguousarray(pe.astype(np.float16))

        with self.namescope('positional_embeddings'):
            pe = tf.constant(pe, dtype=x.dtype)
            pe = self.norm(pe)
            x = self.add(x, pe)
            x = self.dropout(x)
        return x

    def add(self, a, b):
        return tf.add(a, b)

    def namescope(self, debug_string):
        return tf.variable_scope(debug_string, reuse=tf.AUTO_REUSE, use_resource=True)
