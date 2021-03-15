# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
from scipy.stats import truncnorm


class Block(object):
    """ Base class for building blocks of conformer model
    :param popart.builder.Builder builder: Popart Builder object
    :param type dtype: numpy data type to use for weights
    """
    layer_norm_eps: float = 0.001

    def __init__(self, builder, dtype):
        self.builder = builder
        self.dtype = dtype

    def xavier_init(self, shape, num_units_in, num_units_out):
        """ xavier initializer for given tensor shape """
        bound = np.sqrt(6. / (num_units_in + num_units_out))
        return np.random.uniform(-bound, bound, shape).astype(self.dtype)

    def normal_init(self, shape, mean, std_dev):
        """ normal initializer for given tensor shape """
        # Truncated random normal between 2 standard deviations
        data = truncnorm.rvs(-2, 2, loc=mean,
                             scale=std_dev, size=np.prod(shape))
        data = data.reshape(shape).astype(self.dtype)
        return data

    def namescope(self, debug_string):
        return self.builder.nameScope(debug_string)

    def get_constant(self, const_value):
        """ returns constant onnx object with given numpy array value """
        return self.builder.aiOnnx.constant(np.array(const_value).astype(self.dtype))

    def add_tensor(self, var_name, init_weights):
        """ adds an initialized weight tensor to the graph """
        weights_tensor = self.builder.addInitializedInputTensor(init_weights, var_name)
        return weights_tensor

    def layer_norm(self, input_x, num_features):
        """ Applies layer normalization to  input_x """
        gamma = self.get_constant(np.ones((num_features,)))
        beta = self.get_constant(np.zeros((num_features,)))
        # converting to 2d spatial tensor for group-norm to work correctly
        input_x = self.builder.aiOnnx.unsqueeze([input_x], axes=[3])
        outs = self.builder.aiGraphcore.groupnormalization([input_x, gamma, beta],
                                                           num_groups=1, epsilon=self.layer_norm_eps)
        out = self.builder.aiOnnx.squeeze([outs[0]], axes=[3])
        return out

    def batch_norm(self, input_x, num_features):
        """ Applies batch normalization to  input_x """
        init_scale = np.ones([num_features]).astype(self.dtype)
        scale = self.add_tensor("scale", init_scale)

        init_biases = np.zeros([num_features]).astype(self.dtype)
        biases = self.add_tensor("biases", init_biases)

        mean = self.add_tensor("mean", np.zeros([num_features]).astype(self.dtype))
        var = self.add_tensor("var", np.zeros([num_features]).astype(self.dtype))

        (out, *__) = self.builder.aiOnnx.batchnormalization([input_x, scale, biases, mean, var],
                                                            num_outputs=5)
        return out

    def glu_activation(self, input_x):
        """ Applies gated-linear-unit activation to input_x """
        s1, s2 = self.builder.aiOnnx.split([input_x], num_outputs=2, axis=1)
        s2_gated = self.builder.aiOnnx.sigmoid([s2])
        out = self.builder.aiOnnx.mul([s1, s2_gated])
        return out

    def swish_activation(self, input_x):
        """ Applies swish activation to input_x """
        out = self.builder.aiOnnx.mul([input_x, self.builder.aiOnnx.sigmoid([input_x])])
        return out


class Linear(Block):
    """ Linear transformation block
    :param popart.builder.Builder builder: Popart Builder object
    :param int num_in_features: the number of input features
    :param int num_out_features: the number of output features
    :param bool bias: whether to have bias term or not
    :param type dtype: numpy data type to use for weights
    """
    def __init__(self, builder, num_in_features, num_out_features, bias=False, dtype=np.float32):
        super(Linear, self).__init__(builder, dtype)
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.bias = bias

    def __call__(self, x):
        return self.__build_graph(x)

    def __build_graph(self, x):

        with self.namescope("Linear"):

            wshape = [self.num_out_features, self.num_in_features]
            init_weights = self.xavier_init(wshape, self.num_in_features, self.num_out_features)
            weights = self.add_tensor("weights", init_weights)
            out = self.builder.aiOnnx.matmul([weights, x])
            if self.bias:
                bshape = [self.num_out_features, 1]
                init_biases = np.zeros(bshape).astype(self.dtype)
                biases = self.add_tensor("bias", init_biases)
                out = self.builder.aiOnnx.add([out, biases])

        return out


class FeedForwardModule(Block):
    """ Feed Forward Module (Figure 4 of conformer paper)
    :param popart.builder.Builder builder: Popart Builder object
    :param int num_channels: the number of channels
    :param float dropout_rate: rate of dropout
    :param type dtype: numpy data type to use for weights
    """
    def __init__(self, builder, num_channels, dropout_rate=0.05, dtype=np.float32):
        super(FeedForwardModule, self).__init__(builder, dtype)
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate

    def __call__(self, x):
        return self.__build_graph(x)

    def __build_graph(self, x):

        builder = self.builder
        num_channels = self.num_channels
        dropout_rate = self.dropout_rate

        with self.namescope("FeedForwardModule"):

            out = self.layer_norm(x, self.num_channels)
            # first linear transformation (expansion factor of 4)
            out = Linear(builder, num_channels, 4 * num_channels, bias=True, dtype=self.dtype)(out)
            out = self.swish_activation(out)
            out = self.builder.aiOnnx.dropout([out], 1, dropout_rate)[0]
            # second linear transformation (project back to model dimension)
            out = Linear(builder, 4 * num_channels, num_channels, bias=True, dtype=self.dtype)(out)
            out = self.builder.aiOnnx.dropout([out], 1, dropout_rate)[0]

        return out


class ConvolutionSubSampler(Block):
    """ Convolutional layer with subsampling
    :param popart.builder.Builder builder: Popart Builder object
    :param int num_channels: the number of channels
    :param int kernel_size: kernel size for convolution
    :param int subsampling_factor: factor by which to subsample input
    :param bool bias: whether to have bias term or not
    :param type dtype: numpy data type to use for weights
    """

    def __init__(self, builder, num_channels, kernel_size, subsampling_factor, bias=True, dtype=np.float32):
        """ Construct a Convolution SubSampler"""
        super(ConvolutionSubSampler, self).__init__(builder, dtype)
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.subsampling_factor = subsampling_factor
        self.bias = bias

    def __call__(self, x):
        return self.__build_graph(x)

    def __build_graph(self, x):

        with self.namescope("ConvolutionSubSampler"):

            wshape = [self.num_channels, self.num_channels, self.kernel_size]
            init_weights = self.xavier_init(wshape, self.num_channels, self.num_channels)
            weights = self.add_tensor("weights", init_weights)
            pad = int(self.kernel_size / 2)
            conv_args = [x, weights]
            if self.bias:
                bshape = [self.num_channels]
                init_biases = np.zeros(bshape).astype(self.dtype)
                biases = self.add_tensor("bias", init_biases)
                conv_args += [biases]
            out = self.builder.aiOnnx.conv(conv_args,
                                           dilations=[1],
                                           kernel_shape=[self.kernel_size],
                                           strides=[self.subsampling_factor],
                                           pads=[pad, pad])

        return out


class ConvolutionModule(Block):
    """ Convolutional Module of conformer model (Figure 2 of conformer paper)
    :param popart.builder.Builder builder: Popart Builder object
    :param int num_channels: the number of channels
    :param int kernel_size: kernel size for convolution
    :param bool bias: whether to have bias term or not
    :param float dropout_rate: rate of dropout
    :param type dtype: numpy data type to use for weights
    """

    def __init__(self, builder, num_channels, kernel_size, bias=True, dropout_rate=0.05, dtype=np.float32):
        """ Construct a Convolution Module """
        super(ConvolutionModule, self).__init__(builder, dtype)
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.dropout_rate = dropout_rate

    def __call__(self, x):
        return self.__build_graph(x)

    def pointwise_conv(self, x, num_out_channels):

        with self.namescope("PointWiseConv"):

            wshape = [num_out_channels, self.num_channels, 1]
            init_weights = self.xavier_init(wshape, self.num_channels, num_out_channels)
            weights = self.add_tensor("weights", init_weights)
            conv_args = [x, weights]
            if self.bias:
                bshape = [num_out_channels]
                init_biases = np.zeros(bshape).astype(self.dtype)
                biases = self.add_tensor("bias", init_biases)
                conv_args += [biases]
            out = self.builder.aiOnnx.conv(conv_args,
                                           dilations=[1],
                                           kernel_shape=[1],
                                           strides=[1],
                                           pads=[0, 0])
        return out

    def depthwise_conv(self, x):

        with self.namescope("DepthWiseConv"):
            # 2nd dimension should be 1 for depth-wise convolutions
            wshape = [self.num_channels, 1, self.kernel_size]
            init_weights = self.xavier_init(wshape, self.num_channels, self.num_channels)
            weights = self.add_tensor("weights", init_weights)
            pad = int(self.kernel_size / 2)
            conv_args = [x, weights]
            if self.bias:
                bshape = [self.num_channels]
                init_biases = np.zeros(bshape).astype(self.dtype)
                biases = self.add_tensor("bias", init_biases)
                conv_args += [biases]
            out = self.builder.aiOnnx.conv(conv_args,
                                           dilations=[1],
                                           kernel_shape=[self.kernel_size],
                                           strides=[1],
                                           group=self.num_channels,
                                           pads=[pad, pad])

        return out

    def __build_graph(self, x):

        with self.namescope("ConvolutionModule"):

            out = self.layer_norm(x, self.num_channels)
            out = self.pointwise_conv(out, 2 * self.num_channels)
            out = self.glu_activation(out)
            out = self.depthwise_conv(out)
            out = self.batch_norm(out, self.num_channels)
            out = self.swish_activation(out)
            out = self.pointwise_conv(out, self.num_channels)
            out = self.builder.aiOnnx.dropout([out], 1, self.dropout_rate)[0]

        return out


class MultiHeadedAttention(Block):
    """ Multi-Head Attention Block.
    :param popart.builder.Builder builder: Popart Builder object
    :param int num_heads: the number of heads
    :param int num_features: the number of features
    :param int sequence_length: length of sequences
    :param float dropout_rate: dropout rate
    :param type dtype: numpy data type to use for weights
    """
    def __init__(self, builder, num_heads, num_features, sequence_length, dropout_rate=0.05, dtype=np.float32):
        """ Construct an MultiHeadedAttention Block """
        super(MultiHeadedAttention, self).__init__(builder, dtype)
        assert(num_features % num_heads == 0)
        self.num_heads = num_heads
        self.num_features = num_features
        self.sequence_length = self.get_constant(sequence_length)
        self.dropout_rate = dropout_rate

    def __call__(self, query, key, value):
        return self.__build_graph(query, key, value)

    def scaled_dot_product_attention(self, Q, K, V):

        Q_t = self.builder.aiOnnx.transpose([Q], perm=[0, 2, 1])  # Tq X q

        # getting transformed query key dot products (Tq X Tk)
        attention_scores = self.builder.aiOnnx.matmul([Q_t, K])
        attention_scores = self.builder.aiOnnx.softmax([attention_scores], axis=2)

        if self.dropout_rate > 0.0:
            attention_scores = self.builder.aiOnnx.dropout([attention_scores], 1, self.dropout_rate)[0]
        attention_scores = self.builder.aiOnnx.transpose([attention_scores], perm=[0, 2, 1])  # Tk X Tq

        # getting weighted average of value vectors to get context vectors
        context_vectors = self.builder.aiOnnx.matmul([V, attention_scores])  # v X Tq

        # scale by sqrt of sequence-length (as in deep-voice attention block)
        # (this scaling was observed to work well relative to scaling by key-dimension before softmax)
        context_vectors = self.builder.aiOnnx.div([context_vectors,
                                                   self.builder.aiOnnx.sqrt([self.sequence_length])])

        return context_vectors

    def __build_graph(self, queries, keys, values):

        with self.namescope("MultiHeadedAttention"):
            builder = self.builder
            num_heads = self.num_heads
            num_features = self.num_features
            Q = Linear(builder, num_features, num_features, bias=False, dtype=self.dtype)(queries)
            K = Linear(builder, num_features, num_features, bias=False, dtype=self.dtype)(keys)
            V = Linear(builder, num_features, num_features, bias=False, dtype=self.dtype)(values)

            Qs = builder.aiOnnx.split([Q], num_outputs=num_heads, axis=1)
            Ks = builder.aiOnnx.split([K], num_outputs=num_heads, axis=1)
            Vs = builder.aiOnnx.split([V], num_outputs=num_heads, axis=1)

            heads = []
            for Qi, Ki, Vi in zip(Qs, Ks, Vs):
                heads.append(self.scaled_dot_product_attention(Qi, Ki, Vi))

            heads_concat = builder.aiOnnx.concat(heads, axis=1)

            context_vecs = Linear(builder, num_features, num_features, bias=False, dtype=self.dtype)(heads_concat)

        return context_vecs


class MultiHeadedSelfAttentionModule(Block):
    """ Multi-Head Attention Block sandiwched between LayerNorm and Dropout (Figure 3 of conformer paper)
    :param popart.builder.Builder builder: Popart Builder object
    :param int num_heads: the number of heads
    :param int num_features: the number of features
    :param int sequence_length: length of sequences
    :param float dropout_rate: dropout rate
    :param type dtype: numpy data type to use for weights
    """
    def __init__(self, builder, num_heads, num_features, sequence_length, dropout_rate=0.05, dtype=np.float32):
        """ Construct an MultiHeadedAttention Module """
        super(MultiHeadedSelfAttentionModule, self).__init__(builder, dtype)
        assert (num_features % num_heads == 0)
        self.num_heads = num_heads
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate

    def __call__(self, x):
        return self.__build_graph(x)

    def __build_graph(self, x):

        with self.namescope("MultiHeadedSelfAttentionModule"):

            mha = MultiHeadedAttention(self.builder,
                                       self.num_heads,
                                       self.num_features,
                                       self.sequence_length,
                                       dropout_rate=self.dropout_rate,
                                       dtype=self.dtype)

            out = self.layer_norm(x, self.num_features)
            out = mha(out, out, out)
            out = self.builder.aiOnnx.dropout([out], 1, self.dropout_rate)[0]

        return out


class ConformerBlock(Block):
    """ Conformer Block (Figure 1 of conformer paper)
    :param popart.builder.Builder builder: Popart Builder object
    :param int num_heads: the number of attention heads
    :param int num_features: the number of features
    :param int sequence_length: length of sequences
    :param int kernel_size: width of kernel
    :param bool use_conv_module: whether to include convolution module or not
    (if False, reduces to Transformer-lite block)
    :param float dropout_rate: dropout rate
    :param type dtype: numpy data type to use for weights
    """
    def __init__(self, builder, num_heads, num_features, sequence_length, kernel_size=31,
                 use_conv_module=True, dropout_rate=0.05, dtype=np.float32):
        """ Construct a Conformer Block """
        super(ConformerBlock, self).__init__(builder, dtype)
        assert (num_features % num_heads == 0)
        self.num_heads = num_heads
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.kernel_size = kernel_size
        self.use_conv_module = use_conv_module
        self.dropout_rate = dropout_rate

    def __call__(self, x):
        return self.__build_graph(x)

    def __build_graph(self, x):

        with self.namescope("ConformerBlock"):

            half_constant = self.get_constant(0.5)

            # first feed-forward layer
            ffn_1 = FeedForwardModule(self.builder,
                                      self.num_features,
                                      dropout_rate=self.dropout_rate,
                                      dtype=self.dtype)
            # second feed-forward layer
            ffn_2 = FeedForwardModule(self.builder,
                                      self.num_features,
                                      dropout_rate=self.dropout_rate,
                                      dtype=self.dtype)
            # multi-headed self-attention module
            mhsa = MultiHeadedSelfAttentionModule(self.builder,
                                                  self.num_heads,
                                                  self.num_features,
                                                  self.sequence_length,
                                                  dropout_rate=self.dropout_rate,
                                                  dtype=self.dtype)
            if self.use_conv_module:
                # convolution module
                conv_module = ConvolutionModule(self.builder,
                                                self.num_features,
                                                self.kernel_size,
                                                dropout_rate=self.dropout_rate,
                                                dtype=self.dtype)

            out = self.builder.aiOnnx.add([x, self.builder.aiOnnx.mul([ffn_1(x), half_constant])])
            out = self.builder.aiOnnx.add([out, mhsa(out)])
            if self.use_conv_module:
                out = self.builder.aiOnnx.add([out, conv_module(out)])
            out = self.builder.aiOnnx.add([out, self.builder.aiOnnx.mul([ffn_2(out), half_constant])])
            out = self.layer_norm(out, self.num_features)

        return out
