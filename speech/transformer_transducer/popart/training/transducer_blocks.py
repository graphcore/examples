# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import math
import sys


class Block(object):
    """ Base class for building blocks of transducer model with RNNT loss """
    global_param_count = 0  # this is a global parameter count
    layer_norm_eps: float = 0.001

    def __init__(self, builder, dtype, block_name):
        self.builder = builder
        self.dtype = dtype
        self.block_name = block_name
        # a list to keep track of children
        self.child_blocks = []
        self.root_param_count = 0  # param-count of this block (excluding children)
        self.name_to_tensor_map = dict()

    def xavier_init(self, shape, num_units_in, num_units_out):
        """ xavier initializer for given tensor shape """
        bound = np.sqrt(6. / (num_units_in + num_units_out))
        return np.random.uniform(-bound, bound, shape).astype(self.dtype)

    def normal_init(self, shape, mean, std_dev):
        """ normal initializer for given tensor shape """
        data = np.random.normal(mean, std_dev, shape).astype(self.dtype)
        return data

    def uniform_init(self, shape, low, high):
        """ uniform initializer for given tensor shape """
        data = np.random.uniform(low, high, shape).astype(self.dtype)
        return data

    def namescope(self, debug_string):
        return self.builder.nameScope(debug_string)

    def get_constant(self, const_value):
        """ returns constant onnx object with given numpy array value """
        return self.builder.aiOnnx.constant(np.array(const_value).astype(self.dtype))

    def add_tensor(self, var_name, init_weights):
        """ adds an initialized weight tensor to the graph """
        tensor_name = self.block_name + '/' + var_name
        if tensor_name in self.name_to_tensor_map.keys():
            print("Tensor with name {} already exists for block {}".format(var_name, self.block_name))
            sys.exit(-1)
        Block.global_param_count += init_weights.size
        self.root_param_count += init_weights.size
        weights_tensor = self.builder.addInitializedInputTensor(init_weights, var_name)
        self.name_to_tensor_map[tensor_name] = weights_tensor
        return weights_tensor

    def apply_dropout(self, tensor, dropout_rate):
        if dropout_rate > 0.0:
            return self.builder.aiOnnx.dropout([tensor], 1, dropout_rate)[0]
        else:
            return tensor

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

    @property
    def param_count(self):
        return self.root_param_count + sum([c.param_count for c in self.child_blocks])

    @property
    def tensor_list(self):
        # first get tensor list of root-node
        tensor_list = list(self.name_to_tensor_map.items())
        # now extend with tensor lists from child blocks
        for cb in self.child_blocks:
            tensor_list.extend(cb.tensor_list)
        return tensor_list


class EmbeddingBlock(Block):
    """ Embedding layer
    :param int num_symbols: number of symbols to embed
    :param int embedding_dim: dimension of embedding
    """
    def __init__(self, builder, num_symbols, embedding_dim, dtype, block_name):
        super(EmbeddingBlock, self).__init__(builder, dtype, block_name)
        self.num_symbols = num_symbols
        self.embedding_dim = embedding_dim

        # create initialized tensors in __init__
        with self.namescope("EmbeddingBlock"):

            init_weights = self.normal_init((self.num_symbols, self.embedding_dim),
                                            mean=0.0, std_dev=1.0)
            self.embedding_matrix = self.add_tensor("embedding_matrix", init_weights)

    def __call__(self, x):
        return self.__build_graph(x)

    def __build_graph(self, token_sequence):
        # input shape to this layer is assumed to be [batch_size, target_sequence_length]
        with self.namescope("EmbeddingBlock"):

            embeddings = self.builder.aiOnnx.gather([self.embedding_matrix, token_sequence])

        # out shape should be [batch_size, target_sequence_length, embedding_dim]
        return embeddings


class Linear(Block):
    """ Linear transformation block (implements the op W * X + b)
    :param int num_in_features: the number of input features
    :param int num_out_features: the number of output features
    """
    def __init__(self, builder, num_in_features, num_out_features, dtype, block_name, bias=True):
        super(Linear, self).__init__(builder, dtype, block_name)
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.bias = bias
        self.init_mag = 1.0 / math.sqrt(self.num_in_features)

        # create initialized tensors in __init__
        with self.namescope("Linear"):

            wshape = [self.num_out_features, self.num_in_features]
            init_weights = self.uniform_init(wshape, -self.init_mag, self.init_mag)
            self.weights = self.add_tensor("weights", init_weights)
            if self.bias:
                bshape = [self.num_out_features, 1]
                init_biases = self.uniform_init(bshape, -self.init_mag, self.init_mag)
                self.biases = self.add_tensor("bias", init_biases)

    def __call__(self, x, force_recompute=False):
        return self.__build_graph(x, force_recompute)

    def __build_graph(self, x, force_recompute):

        with self.namescope("Linear"):

            out = self.builder.aiOnnx.matmul([self.weights, x])
            if force_recompute:
                self.builder.recomputeOutputInBackwardPass(out)
            if self.bias:
                out = self.builder.aiOnnx.add([out, self.biases])
                if force_recompute:
                    self.builder.recomputeOutputInBackwardPass(out)

        return out


class RHSLinear(Block):
    """ Linear transformation block (implements the op X * W + b)
    :param int num_in_features: the number of input features
    :param int num_out_features: the number of output features
    """
    def __init__(self, builder, num_in_features, num_out_features, dtype, block_name, bias=True):
        super(RHSLinear, self).__init__(builder, dtype, block_name)
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.bias = bias
        self.init_mag = 1.0 / math.sqrt(self.num_in_features)

        # create initialized tensors in __init__
        with self.namescope("RHSLinear"):

            wshape = [self.num_in_features, self.num_out_features]
            init_weights = self.uniform_init(wshape, -self.init_mag, self.init_mag)
            self.weights = self.add_tensor("weights", init_weights)
            if self.bias:
                bshape = [1, self.num_out_features]
                init_biases = self.uniform_init(bshape, -self.init_mag, self.init_mag)
                self.biases = self.add_tensor("bias", init_biases)

    def __call__(self, x, force_recompute=False):
        return self.__build_graph(x, force_recompute)

    def __build_graph(self, x, force_recompute):

        with self.namescope("RHSLinear"):

            out = self.builder.aiOnnx.matmul([x, self.weights])
            if force_recompute:
                self.builder.recomputeOutputInBackwardPass(out)
            if self.bias:
                out = self.builder.aiOnnx.add([out, self.biases])
                if force_recompute:
                    self.builder.recomputeOutputInBackwardPass(out)

        return out


class ConvolutionSubSampler(Block):
    """ Convolutional layer with subsampling
    :param int num_in_channels: the number of input channels
    :param int num_out_channels: the number of output channels
    :param int kernel_size: kernel size for convolution
    :param int subsampling_factor: factor by which to subsample input
    """
    def __init__(self, builder, num_in_channels, num_out_channels, kernel_size, subsampling_factor, dtype, block_name, bias=True):
        """ Construct a Convolution SubSampler"""
        super(ConvolutionSubSampler, self).__init__(builder, dtype, block_name)
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.kernel_size = kernel_size
        self.subsampling_factor = subsampling_factor
        self.bias = bias

        with self.namescope("ConvolutionSubSampler"):

            wshape = [self.num_out_channels, self.num_in_channels, self.kernel_size]
            init_weights = self.xavier_init(wshape, self.num_in_channels, self.num_out_channels)
            self.weights = self.add_tensor("weights", init_weights)
            if self.bias:
                bshape = [self.num_out_channels]
                init_biases = np.zeros(bshape).astype(self.dtype)
                self.biases = self.add_tensor("bias", init_biases)

    def __call__(self, x):
        return self.__build_graph(x)

    def __build_graph(self, x):

        with self.namescope("ConvolutionSubSampler"):

            pad = self.kernel_size - 1
            conv_args = [x, self.weights]
            if self.bias:
                conv_args += [self.biases]
            out = self.builder.aiOnnx.conv(conv_args,
                                           dilations=[1],
                                           kernel_shape=[self.kernel_size],
                                           strides=[self.subsampling_factor],
                                           pads=[pad, 0])

        return out


class MultiHeadedAttention(Block):
    """ Multi-Head Attention Block.
    :param int num_heads: the number of heads
    :param int num_features: the number of features
    :param float dropout_rate: dropout rate
    """
    def __init__(self, builder, num_heads, num_features, dtype, block_name, dropout_rate=0.05):
        """ Construct an MultiHeadedAttention Block """
        super(MultiHeadedAttention, self).__init__(builder, dtype, block_name)
        assert(num_features % num_heads == 0)
        self.num_heads = num_heads
        self.num_features = num_features
        self.dropout_rate = dropout_rate

        with self.namescope("MultiHeadedAttention"):
            self.Q_proj = Linear(builder, num_features, num_features, dtype, block_name + "/Q_proj", bias=False)
            self.K_proj = Linear(builder, num_features, num_features, dtype, block_name + "/K_proj", bias=False)
            self.V_proj = Linear(builder, num_features, num_features, dtype, block_name + "/V_proj", bias=False)

            self.out_proj = Linear(builder, num_features, num_features, dtype, block_name + "/out_proj", bias=False)

            self.child_blocks = [self.Q_proj, self.K_proj, self.V_proj, self.out_proj]

    def __call__(self, query, key, value, force_recompute=False):
        return self.__build_graph(query, key, value, force_recompute=force_recompute)

    def scaled_dot_product_attention(self, Q, K, V, inv_sqrt_qkv_dim):

        Q_t = self.builder.aiOnnx.transpose([Q], perm=[0, 2, 1])  # Tq X q

        # getting transformed query key dot products (Tq X Tk)
        attention_scores = self.builder.aiOnnx.matmul([Q_t, K])
        attention_scores = self.builder.aiOnnx.mul([attention_scores, inv_sqrt_qkv_dim])
        attention_scores = self.builder.aiOnnx.softmax([attention_scores], axis=2)

        if self.dropout_rate > 0.0:
            attention_scores = self.apply_dropout(attention_scores, self.dropout_rate)
        attention_scores = self.builder.aiOnnx.transpose([attention_scores], perm=[0, 2, 1])  # Tk X Tq

        # getting weighted average of value vectors to get context vectors
        context_vectors = self.builder.aiOnnx.matmul([V, attention_scores])  # v X Tq

        return context_vectors

    def __build_graph(self, queries, keys, values, force_recompute):

        with self.namescope("MultiHeadedAttention"):

            builder = self.builder
            num_heads = self.num_heads

            Q = self.Q_proj(queries, force_recompute=force_recompute)
            K = self.K_proj(keys, force_recompute=force_recompute)
            V = self.V_proj(values, force_recompute=force_recompute)

            Qs = builder.aiOnnx.split([Q], num_outputs=num_heads, axis=1)
            Ks = builder.aiOnnx.split([K], num_outputs=num_heads, axis=1)
            Vs = builder.aiOnnx.split([V], num_outputs=num_heads, axis=1)

            inv_sqrt_qkv_dim = self.get_constant(1.0 / np.sqrt(self.num_features // self.num_heads))

            heads = []
            for Qi, Ki, Vi in zip(Qs, Ks, Vs):
                heads.append(self.scaled_dot_product_attention(Qi, Ki, Vi, inv_sqrt_qkv_dim))

            heads_concat = builder.aiOnnx.concat(heads, axis=1)

            context_vecs = self.out_proj(heads_concat, force_recompute=force_recompute)

        return context_vecs


class TransformerBlock(Block):
    """ Transformer Block
    :param int num_heads: the number of attention heads
    :param int num_features: the number of features
    :param float dropout_rate: dropout rate
    """
    def __init__(self, builder, num_heads, num_features, dtype, block_name, dropout_rate=0.05):
        """ Construct a Transformer Block """
        super(TransformerBlock, self).__init__(builder, dtype, block_name)
        assert (num_features % num_heads == 0)
        self.num_heads = num_heads
        self.num_features = num_features
        self.dropout_rate = dropout_rate

        with self.namescope("TransformerBlock"):

            self.mhsa = MultiHeadedAttention(builder,
                                             num_heads,
                                             num_features,
                                             dtype,
                                             block_name + "/mha_block",
                                             dropout_rate=dropout_rate)

            # linear layers for FFN
            self.linear_1 = Linear(builder, num_features, 4 * num_features, dtype, block_name + "/Linear1")
            self.linear_2 = Linear(builder, 4 * num_features, num_features, dtype, block_name + "/Linear2")

            self.child_blocks = [self.mhsa, self.linear_1, self.linear_2]

    def __call__(self, x, force_recompute=False):
        return self.__build_graph(x, force_recompute=force_recompute)

    def __build_graph(self, x, force_recompute):

        with self.namescope("TransformerBlock"):

            out = self.builder.aiOnnx.add([x, self.mhsa(x, x, x, force_recompute=force_recompute)])
            if force_recompute:
                self.builder.recomputeOutputInBackwardPass(out)
            out = self.layer_norm(out, self.num_features)
            if force_recompute:
                self.builder.recomputeOutputInBackwardPass(out)

            ffn_out = self.linear_1(out, force_recompute=force_recompute)
            ffn_out = self.builder.aiOnnx.relu([ffn_out])
            if force_recompute:
                self.builder.recomputeOutputInBackwardPass(ffn_out)
            ffn_out = self.apply_dropout(ffn_out, self.dropout_rate)
            if force_recompute:
                self.builder.recomputeOutputInBackwardPass(ffn_out)
            ffn_out = self.linear_2(ffn_out, force_recompute=True)

            out = self.builder.aiOnnx.add([out, ffn_out])
            if force_recompute:
                self.builder.recomputeOutputInBackwardPass(out)
            out = self.layer_norm(out, self.num_features)
            if force_recompute:
                self.builder.recomputeOutputInBackwardPass(out)

        return out


class Split(Block):
    """ Splits input based on given split size and split_axis
    :param int total_size: size of tensor to split along split_axis
    :param int split_size: the size of each input split
    :param int split_axis: axis along which to perform split
    """
    def __init__(self, builder, total_size, split_size, split_axis, dtype, block_name):
        super(Split, self).__init__(builder, dtype, block_name)
        self.total_size = total_size
        self.split_size = split_size
        self.split_axis = split_axis

    def __call__(self, x):
        return self.__build_graph(x)

    def __build_graph(self, x):

        with self.namescope("Split"):

            builder = self.builder

            split_size = self.split_size
            num_splits = self.total_size // split_size
            split_rem = self.total_size - num_splits * split_size
            if split_rem > 0:
                split_out = [split_size] * num_splits + [split_rem]
                num_splits += 1
            else:
                split_out = [split_size] * num_splits

            x_splits = builder.aiOnnx.split([x],
                                            num_outputs=len(split_out),
                                            axis=self.split_axis,
                                            split=split_out)

        return x_splits


class LSTM(Block):
    """ LSTM layer
    :param int num_in_features: the number of input features
    :param int num_hidden_features: the number of hidden features
    """
    def __init__(self, builder, num_in_features, num_hidden_features, dtype, block_name,
                 forget_gate_bias=None, weights_init_scale=1.0):
        super(LSTM, self).__init__(builder, dtype, block_name)
        self.num_in_features = num_in_features
        self.num_hidden_features = num_hidden_features
        self.init_mag = 1.0 / math.sqrt(self.num_hidden_features)

        # create initialized tensors in __init__
        with self.namescope("LSTM"):

            # this ordering of weights is consistent with the definition of weights in the ONNX specification
            lstm_weight_dict = {}
            for wname in ['w_i', 'w_o', 'w_f', 'w_c']:
                lstm_weight_dict[wname] = self.uniform_init([1, self.num_hidden_features, self.num_in_features],
                                                            -self.init_mag, self.init_mag) * weights_init_scale
            for wname in ['r_i', 'r_o', 'r_f', 'r_c']:
                lstm_weight_dict[wname] = self.uniform_init([1, self.num_hidden_features, self.num_hidden_features],
                                                            -self.init_mag, self.init_mag) * weights_init_scale

            # lstm_input_weights_i - this is the same as weight tensor W in the ONNX specification
            lstm_input_weights_i = np.concatenate([lstm_weight_dict[wname] for wname in ['w_i', 'w_o', 'w_f', 'w_c']],
                                                  axis=1)
            # lstm_output_weights_i - this is the same as recurrence weight tensor R in the ONNX specification
            lstm_output_weights_i = np.concatenate([lstm_weight_dict[wname] for wname in ['r_i', 'r_o', 'r_f', 'r_c']],
                                                   axis=1)
            self.lstm_input_weights = self.add_tensor("lstm_input_weights", lstm_input_weights_i)
            self.lstm_output_weights = self.add_tensor("lstm_output_weights", lstm_output_weights_i)

            lstm_bias_names = ['wb_i', 'wb_o', 'wb_f', 'wb_c', 'rb_i', 'rb_o', 'rb_f', 'rb_c']
            lstm_bias_dict = {}
            for bname in lstm_bias_names:
                bias_init_value = self.uniform_init([1, self.num_hidden_features],
                                                    -self.init_mag, self.init_mag)
                if forget_gate_bias is not None:
                    # set forget gate biases so that they add up to forget_gate_bias for all features
                    if bname == 'wb_f':
                        bias_init_value = np.ones([1, self.num_hidden_features]).astype(self.dtype) * forget_gate_bias
                    elif bname == 'rb_f':
                        bias_init_value = np.zeros([1, self.num_hidden_features]).astype(self.dtype)
                lstm_bias_dict[bname] = bias_init_value * weights_init_scale
            # lstm_biases_i - this is the same as bias tensor B in the ONNX specification
            lstm_biases_i = np.concatenate([lstm_bias_dict[bname] for bname in lstm_bias_names], axis=1)
            self.lstm_biases = self.add_tensor("lstm_biases", lstm_biases_i)

    def __call__(self, x, x_lens=None, force_recompute=False):
        return self.__build_graph(x, x_lens, force_recompute)

    def __build_graph(self, x, x_lens, force_recompute):
        # input shape to lstm must be [seq_length, batch_size, channel_dim]

        with self.namescope("LSTM"):

            lstm_args = [x, self.lstm_input_weights, self.lstm_output_weights, self.lstm_biases]
            if x_lens:
                lstm_args.append(x_lens)

            lstm_outputs = self.builder.aiOnnx.lstm(lstm_args,
                                                    num_outputs=3, clip=None, hidden_size=self.num_hidden_features)
            if force_recompute:
                self.builder.recomputeOutputInBackwardPass(set(lstm_outputs))
            out = lstm_outputs[0]

            # squeeze out num-directions dimension
            out = self.builder.aiOnnx.squeeze([out], axes=[1])
            if force_recompute:
                self.builder.recomputeOutputInBackwardPass(out)

        return out
