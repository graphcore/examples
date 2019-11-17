# Copyright 2019 Graphcore Ltd.
import numpy as np
import tensorflow as tf

from ..transformer_base import Transformer


class TensorflowTransformer(Transformer):
    def __init__(self, opts, *args, **kwargs):
        self.dtype = np.float16
        self.encoder_k = None
        self.encoder_v = None
        super(TensorflowTransformer, self).__init__(opts, *args, **kwargs)

    def _get_variable(self, name, shape, init=None):
        if init is None:
            init = tf.constant_initializer(0, dtype=self.dtype)
        return tf.get_variable(name, shape, initializer=init, dtype=self.dtype)

    def gemm(self, x, w, b):
        return tf.matmul(x, w) + b

    # x[batch_size*sequence_len] -> x[batch_size*sequence_len, embedding_len]
    def embedding(self, x, src, debug_name=''):
        vocab_len = self.src_vocab_len if src else self.tgt_vocab_len
        with self.namescope('{}embedding'.format(debug_name)):
            # TODO: allow for embedding reuse/tying.
            embedding_dict = self._get_variable(
                "dict", [vocab_len, self.embedding_len])
            x = tf.gather(embedding_dict, x)

            # multiply by sqrt(d_model)
            c0 = tf.constant(
                np.sqrt(np.array([self.embedding_len]).astype(self.dtype)))
            x = x * c0
        return x

    # x[..,embedding_len] -> x[..,tgt_vocab_len]
    def projection(self, x):
        with self.namescope('projection'):
            decoder_w = self._get_variable(
                'decoder_w', [self.embedding_len, self.tgt_vocab_len])
            if self.projection_bias:
                decoder_b = self._get_variable(
                    'decoder_b', [self.tgt_vocab_len])
                x = self.gemm(x, decoder_w, decoder_b)
            else:
                x = tf.matmul(x, decoder_w)

            x = tf.reshape(
                x, [self.sequence_len, self.batch_size, self.tgt_vocab_len])
        return x

    # -> x
    def add(self, x, y):
        return y + x

    # -> x
    def norm(self, x):
        with self.namescope('layernorm'):
            x = tf.contrib.layers.layer_norm(x)
        return x

    # -> x
    def feed_forward(self, x):
        with self.namescope('ffn'):
            inner_dim = self.ff_len

            with self.namescope('1'):
                w1 = self._get_variable('w1', [self.embedding_len, inner_dim])
                b1 = self._get_variable('b1', [inner_dim])
                x = self.gemm(x, w1, b1)
            x = tf.nn.relu(x)
            x = self.dropout(x)
            with self.namescope('2'):
                w2 = self._get_variable('w2', [inner_dim, self.embedding_len])
                b2 = self._get_variable('b2', [self.embedding_len])
                x = self.gemm(x, w2, b2)
        return x

    # GEMM/matmul with variable allocation
    def linear(self, x, w_shape, b_shape):
        w = self._get_variable('w', w_shape)
        x = tf.matmul(x, w)
        if self.biases:
            b = self._get_variable('b', b_shape)
            x = x + b
        return x

    def attention(self, in_q, in_k, in_v, mask=None, debug_name=''):
        with self.namescope(debug_name):
            bs = self.batch_size
            sl = self.sequence_len
            el = self.embedding_len
            ah = self.attention_heads
            w_shape = [el, ah * self.qkv_len]
            b_shape = [ah * self.qkv_len]

            # self_attn
            if self.grouped_attention and in_q == in_k == in_v:
                w_shape[1] *= 3
                b_shape[0] *= 3
                with self.namescope('qkv'):
                    qkv = self.linear(in_q, w_shape, b_shape)
                    qkv = tf.reshape(qkv, [bs * sl, 3, ah * self.qkv_len])
                    qkv = tf.transpose(qkv, [1, 0, 2])
                    q, k, v = [qkv[i:i+1] for i in range(3)]
            else:
                # Q
                with self.namescope('q'):
                    q = self.linear(in_q, w_shape, b_shape)

                # Encoder attention. Reuse previous calculation if possible.
                if self.shared_encoder_attention and in_q != in_k:
                    with self.namescope('k'):
                        if self.encoder_k is None:
                            k = self.linear(in_k, w_shape, b_shape)
                        else:
                            k = self.encoder_k
                    with self.namescope('v'):
                        if self.encoder_v is None:
                            v = self.linear(in_v, w_shape, b_shape)
                        else:
                            v = self.encoder_v
                    self.encoder_k, self.encoder_v = (k, v)
                else:
                    # K
                    with self.namescope('k'):
                        k = self.linear(in_k, w_shape, b_shape)
                    # V
                    with self.namescope('v'):
                        v = self.linear(in_v, w_shape, b_shape)

            q, k, v = [
                tf.transpose(tf.reshape(qkv, [bs * sl, ah, self.qkv_len]),
                             [1, 0, 2])
                for qkv in [q, k, v]]

            # Attention calculation
            with self.namescope('z'):
                x = tf.matmul(q, k, transpose_b=True)

                # This was 0.1249999925494194
                c = tf.constant(
                    1/np.sqrt(np.array([self.qkv_len]).astype(self.dtype)))
                x = x * c

                if mask is not None:
                    # NOTE: No where op. GoogleResearch's BERT implementation uses add for the mask.
                    x = x + mask

                x = tf.nn.softmax(x, axis=0)

                z = tf.matmul(x, v)

                z = tf.reshape(z, [bs * sl, ah * self.qkv_len])

            # Project back to embedding_len
            with self.namescope('out'):
                w = self._get_variable('w', [ah * self.qkv_len, el])
                if self.biases:
                    b = self._get_variable('b', [el])
                    z = self.gemm(z, w, b)
                else:
                    z = tf.matmul(z, w)

        return z

    def position_encoder(self, x, debug_name=''):  # -> x
        # TODO: is this shared between encoder & decoder?

        pe = np.zeros([self.sequence_len, self.embedding_len], self.dtype)

        # pe = sin(pos/(10000**(2i/d_model))) or cos(pos/10000**((2i+1)/d_model))
        # pos = sequence dimension index. i embedding dimension index.
        d_model = self.embedding_len
        pos = np.arange(0, self.sequence_len).reshape(-1, 1)
        i = np.arange(0, d_model, 2).reshape(1, -1)

        # Calculate div_term in log space.
        # TODO: Check this against paper
        div_term = np.exp(-(np.log(10000.0) * i / d_model))

        # Even dims sin, odd dims cos.
        # TODO: Could be sin where pos < sequence_len/2 & cos
        #    where pos >= sequence_len/2?
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)

        # Add batch dimension
        pe = np.broadcast_to(
            pe, [self.batch_size, self.sequence_len, self.embedding_len])
        # Dim shuffle to time major
        pe = np.swapaxes(pe, 0, 1)
        pe = np.reshape(
            pe, [self.batch_size * self.sequence_len, self.embedding_len])

        with self.namescope('{}positional_encoder'.format(debug_name)):
            # TODO: Used as constant but change to a variable if needs to be trained.
            pe = tf.constant(pe)

            x = x + pe
            x = self.dropout(x)
        return x

    def dropout(self, x):  # -> x
        # TODO: Needs poplibs implementation
        return x

    def namescope(self, debug_string):
        return tf.variable_scope(debug_string)
