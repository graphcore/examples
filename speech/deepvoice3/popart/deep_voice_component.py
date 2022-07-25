# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import sys
import numpy as np
from scipy.stats import truncnorm
import logging_util

# set up logging
logger = logging_util.get_basic_logger(__name__)


class Component(object):
    """ base class for components of deep-voice model (encoder, decoder and converter) """
    def __init__(self, conf, builder, conv_type="causal", graph_initial_weights=None, graph_name_to_tensor_map=None):

        self.builder = builder
        self.dtype = conf.precision
        self.conf = conf
        self.conv_type = conv_type
        self.init_type = conf.init_type
        self.graph_initial_weights = graph_initial_weights
        self.graph_name_to_tensor_map = graph_name_to_tensor_map

    def xavier_init(self, shape, num_units_in, num_units_out):
        bound = np.sqrt(6. / (num_units_in + num_units_out))
        return np.random.uniform(-bound, bound, shape).astype(self.dtype)

    def normal_init(self, shape, mean, std_dev):
        # Truncated random normal between 2 standard deviations
        data = truncnorm.rvs(-2, 2, loc=mean,
                             scale=std_dev, size=np.prod(shape))
        data = data.reshape(shape).astype(self.dtype)
        return data

    def namescope(self, debug_string):
        return self.builder.nameScope(debug_string)

    def get_constant(self, const_value):
        return self.builder.aiOnnx.constant(np.array(const_value).astype(self.dtype))

    def add_tensor(self, block_name, var_name, init_weights):
        """ adds an initialized weight tensor to the graph for given block and variable """
        tensor_name = block_name + '/' + var_name
        if self.graph_initial_weights is not None:
            if tensor_name in self.name_to_tensor.keys():
                # return tensor if it has already been created (for auto-regression)
                return self.name_to_tensor[tensor_name]
            else:
                # using weights from trained model
                init_weights = self.graph_initial_weights[self.graph_name_to_tensor_map[tensor_name]]
        else:
            if tensor_name in self.name_to_tensor.keys():
                logger.error("Tensor with name {} already exists".format(tensor_name))
                sys.exit(-1)
        weights_tensor = self.builder.addInitializedInputTensor(init_weights, var_name)
        self.name_to_tensor[tensor_name] = weights_tensor
        return weights_tensor

    def gated_residual_conv_block(self, x, num_channels, block_name, speaker_embedding=None, speaker_embedding_dim=0,
                                  ksize=5, bias=True, dropout_rate=0.05):
        """ this is the implementation of the conv block (gated linear unit + residual connection)
        described in figure 2 of the deep voice 3 paper """

        with self.namescope("conv_block"):

            gated_conv_out = self.builder.aiOnnx.unsqueeze([x], axes=[3])

            if dropout_rate > 0.0:
                gated_conv_out = self.builder.aiOnnx.dropout([gated_conv_out], 1, dropout_rate)[0]

            wshape = [2 * num_channels, num_channels, ksize, 1]
            if self.init_type == 'xavier':
                init_weights = self.xavier_init(wshape, num_channels, 2 * num_channels)
            elif self.init_type == 'normal':
                init_weights = self.normal_init(wshape, mean=0.0, std_dev=self.conf.layer_normal_init_weights_std_dev)
            else:
                raise ValueError('Not a valid initialization type for conv block:{}'.format(self.init_type))

            weights = self.add_tensor(block_name, "weights", init_weights)

            if self.conv_type == "causal":
                pad = ksize - 1
                pads = [pad, 0, 0, 0]
            elif self.conv_type == "same":
                pad = int(ksize / 2)
                pads = [pad, 0, pad, 0]
            else:
                logger.error("Not a valid padding options for this conv_block. Use same or causal")
                sys.exit(-1)

            conv_args = [gated_conv_out, weights]
            if bias:
                bshape = [2 * num_channels]
                init_biases = np.zeros(bshape).astype(self.dtype)
                biases = self.add_tensor(block_name, "bias", init_biases)
                conv_args += [biases]

            gated_conv_out = self.builder.aiOnnx.conv(conv_args,
                                                      dilations=[1, 1],
                                                      kernel_shape=[ksize, 1],
                                                      strides=[1, 1],
                                                      pads=pads)

            xs1, xs2 = self.builder.aiOnnx.split([self.builder.aiOnnx.squeeze([gated_conv_out], axes=[3])],
                                                 num_outputs=2, axis=1)
            if speaker_embedding:
                xs1 = self.apply_speaker_embedding(xs1, num_channels, speaker_embedding, speaker_embedding_dim,
                                                   block_name + "_apply_speaker_embedding")
            xs2_gated = self.builder.aiOnnx.sigmoid([xs2])

            # gated conv output
            gated_conv_out = self.builder.aiOnnx.mul([xs1, xs2_gated])

            # making the residual connection
            x = self.builder.aiOnnx.mul([self.get_constant(np.sqrt(0.5)),
                                         self.builder.aiOnnx.add([x, gated_conv_out])])

        return x

    def temp_distributed_FC(self, x, channels_in, channels_out, block_name,
                            bias=True, activation=None, given_init_weights=None):
        """ temporally distributed fully-connected layer """

        with self.namescope("temp_dist_fc"):

            wshape = [channels_out, channels_in]
            if given_init_weights is None:
                if self.init_type == 'xavier':
                    init_weights = self.xavier_init(wshape, channels_in, channels_out)
                elif self.init_type == 'normal':
                    init_weights = self.normal_init(wshape, mean=0.0, std_dev=self.conf.layer_normal_init_weights_std_dev)
                else:
                    raise ValueError('Not a valid initialization type for '
                                     'temporally distributed FC block:{}'.format(self.init_type))
            else:
                init_weights = given_init_weights
                assert(list(init_weights.shape) == list(wshape))

            weights = self.add_tensor(block_name, "weights", init_weights)

            x = self.builder.aiOnnx.matmul([weights, x])

            if bias:
                bshape = [channels_out, 1]
                init_biases = np.zeros(bshape).astype(self.dtype)
                biases = self.add_tensor(block_name, "bias", init_biases)
                x = self.builder.aiOnnx.add([x, biases])

            if activation == "relu":
                x = self.builder.aiOnnx.relu([x])
            elif activation == "sigmoid":
                x = self.builder.aiOnnx.sigmoid([x])

        return x

    def attention_block(self, h_k, h_v, h_q, k_dim, v_dim, q_dim, text_seq_length, attention_hidden_size, block_name,
                        attention_dropout_rate=0.05,
                        keys_positional_encodings=None, queries_positional_encodings=None,
                        same_init_query_key_projection=True):
        """ this is the implementation of the attention block described in figure 3 of the deep voice 3 paper """

        # making popart constant for sequence length
        text_seq_length = self.get_constant(text_seq_length)

        if keys_positional_encodings:
            h_k = self.builder.aiOnnx.add([h_k, keys_positional_encodings])
        if queries_positional_encodings:
            h_q = self.builder.aiOnnx.add([h_q, queries_positional_encodings])

        with self.namescope("attention_block"):
            # using temporally distributed FC layers to get transformed keys, values & queries
            if same_init_query_key_projection:
                init_query_key_projection = self.xavier_init([attention_hidden_size, q_dim], q_dim, attention_hidden_size)
            else:
                init_query_key_projection = None
            Q_k = self.temp_distributed_FC(h_k, k_dim, attention_hidden_size, block_name + "_key_projection",
                                           bias=False, given_init_weights=init_query_key_projection)  # k X Tk
            Q_v = self.temp_distributed_FC(h_v, v_dim, attention_hidden_size, block_name + "_value_projection",
                                           bias=False)  # v X Tk
            Q_q = self.temp_distributed_FC(h_q, q_dim, attention_hidden_size, block_name + "_query_projection",
                                           bias=False, given_init_weights=init_query_key_projection)

            # transposing Q_q
            Q_q_t = self.builder.aiOnnx.transpose([Q_q], perm=[0, 2, 1])  # Tq X q

            # getting transformed query key dot products (Tq X Tk)
            attention_scores = self.builder.aiOnnx.matmul([Q_q_t, Q_k])
            attention_scores = self.builder.aiOnnx.softmax([attention_scores], axis=2)

            if attention_dropout_rate > 0.0:
                attention_scores = self.builder.aiOnnx.dropout([attention_scores], 1, attention_dropout_rate)[0]
            attention_scores = self.builder.aiOnnx.transpose([attention_scores], perm=[0, 2, 1])  # Tk X Tq
            self.builder.setInplacePreferences(attention_scores, {"TransposeInplace": 1000.0})

            # getting weighted average of value vectors to get context vectors
            context_vectors = self.builder.aiOnnx.matmul([Q_v, attention_scores])  # v X Tq

            # dividing by sqrt of num-steps
            context_vectors = self.builder.aiOnnx.div([context_vectors, self.builder.aiOnnx.sqrt([text_seq_length])])

            # projecting context vectors back to space with dimension of original queries
            context_vectors = self.temp_distributed_FC(context_vectors, attention_hidden_size, q_dim,
                                                       block_name + "_context_vec_projection",
                                                       activation="relu", bias=False)

        return context_vectors, attention_scores

    def embedding(self, indices, num_symbols, embedding_size, block_name):
        """ Embedding layer """

        with self.namescope("embedding_layer"):

            init_weights = self.normal_init((num_symbols, embedding_size),
                                            mean=0.0, std_dev=self.conf.embed_normal_init_weights_std_dev)
            embedding_matrix = self.add_tensor(block_name, "embedding_matrix", init_weights)

            embeddings = self.builder.aiOnnx.gather([embedding_matrix, indices])

            # converting to NCW format
            embeddings = self.builder.aiOnnx.transpose([embeddings], perm=[0, 2, 1])

        return embeddings, embedding_matrix

    def apply_speaker_embedding(self, input_sequence, num_channels,
                                speaker_embedding, speaker_embedding_dim,
                                block_name):
        """ add speaker embedding to input sequence """

        with self.namescope("speaker_embedding"):

            wshape = [num_channels, speaker_embedding_dim]
            if self.init_type == 'xavier':
                init_weights = self.xavier_init(wshape, speaker_embedding_dim, num_channels)
            elif self.init_type == 'normal':
                init_weights = self.normal_init(wshape, mean=0.0, std_dev=self.conf.layer_normal_init_weights_std_dev)
            else:
                raise ValueError('Not a valid initialization type for '
                                 'speaker embedding application:{}'.format(self.init_type))

            weights = self.add_tensor(block_name, "weights", init_weights)
            bshape = [num_channels, 1]
            init_biases = np.zeros(bshape).astype(self.dtype)
            biases = self.add_tensor(block_name, "bias", init_biases)

            embedding_transformed = \
                self.builder.aiOnnx.tanh([self.builder.aiOnnx.add(
                    [self.builder.aiOnnx.matmul([weights, speaker_embedding]), biases])])

            out_sequence = self.builder.aiOnnx.add([input_sequence, embedding_transformed])

        return out_sequence
