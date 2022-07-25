# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import logging_util

from deep_voice_component import Component

# set up logging
logger = logging_util.get_basic_logger(__name__)


class Encoder(Component):
    def __init__(self, conf, builder, graph_initial_weights=None, graph_name_to_tensor_map=None):
        self.num_symbols = conf.num_symbols
        self.encoder_channels = conf.encoder_channels
        self.speaker_embedding_dim = conf.speaker_embedding_dim
        self.character_embedding_dim = conf.character_embedding_dim
        self.num_speakers = conf.num_speakers
        self.num_encoder_conv_blocks = conf.num_encoder_conv_blocks
        self.dropout_rate = conf.dropout_rate
        super(Encoder, self).__init__(conf, builder, conv_type="same",
                                      graph_initial_weights=graph_initial_weights,
                                      graph_name_to_tensor_map=graph_name_to_tensor_map)

    def __call__(self, x_text, speaker_embedding, name_to_tensor):
        self.name_to_tensor = name_to_tensor
        return self.__build_graph(x_text, speaker_embedding)

    def __build_graph(self, x_text, speaker_embedding):

        logger.info("Building Encoder Graph")

        # embedding layer
        with self.namescope("text_embedding"):
            h_e, character_embedding_matrix = self.embedding(x_text, self.num_symbols, self.character_embedding_dim,
                                                             "text_embedding")
            if speaker_embedding:
                h_e = self.apply_speaker_embedding(h_e, self.character_embedding_dim,
                                                   speaker_embedding, self.speaker_embedding_dim,
                                                   "encoder_pre_apply_speaker_embedding")

        # encoder prenet
        with self.namescope("encoder_prenet"):
            x = self.temp_distributed_FC(h_e, self.character_embedding_dim, self.encoder_channels,
                                         "encoder_prenet", activation="relu")

        with self.namescope("encoder_conv_blocks"):
            for block_ind in range(self.num_encoder_conv_blocks):
                block_name = "encoder_conv_block_" + str(block_ind)
                x = self.gated_residual_conv_block(x, self.encoder_channels, block_name,
                                                   speaker_embedding, self.speaker_embedding_dim,
                                                   dropout_rate=self.dropout_rate)
        # encoder postnet
        with self.namescope("encoder_postnet"):
            # attention key vectors
            h_k = self.temp_distributed_FC(x, self.encoder_channels, self.character_embedding_dim,
                                           "encoder_postnet", activation="relu")

            if speaker_embedding:
                h_k = self.apply_speaker_embedding(h_k, self.character_embedding_dim,
                                                   speaker_embedding, self.speaker_embedding_dim,
                                                   "encoder_post_apply_speaker_embedding")
            # attention value vectors
            h_v = self.builder.aiOnnx.mul([self.get_constant(np.sqrt(0.5)),
                                           self.builder.aiOnnx.add([h_e, h_k])])
        return h_k, h_v


class Decoder(Component):
    def __init__(self, conf, builder, graph_initial_weights=None, graph_name_to_tensor_map=None, for_inference=False):
        self.character_embedding_dim = conf.character_embedding_dim
        self.max_text_sequence_length = conf.max_text_sequence_length
        self.speaker_embedding_dim = conf.speaker_embedding_dim
        self.decoder_channels = conf.decoder_channels
        self.num_decoder_conv_blocks = conf.num_decoder_conv_blocks
        self.decoder_attention_flags = conf.decoder_attention_flags
        self.dropout_rate = conf.dropout_rate
        self.decoder_prenet_sizes = conf.decoder_prenet_sizes
        self.attention_hidden_size = conf.attention_hidden_size
        self.mel_bands = conf.mel_bands
        self.n_frames_per_pred = conf.n_frames_per_pred
        self.num_speakers = conf.num_speakers
        self.for_inference = for_inference
        super(Decoder, self).__init__(conf, builder, conv_type="causal",
                                      graph_initial_weights=graph_initial_weights,
                                      graph_name_to_tensor_map=graph_name_to_tensor_map)

    def __call__(self, h_k, h_v, x_spectrogram, speaker_embedding, name_to_tensor):
        self.name_to_tensor = name_to_tensor
        if not self.for_inference:
            return self.__build_graph(h_k, h_v, x_spectrogram, speaker_embedding)
        else:
            return self.__build_graph_for_inference(h_k, h_v, speaker_embedding)

    def __build_graph(self, h_k, h_v, x_spectrogram, speaker_embedding):

        logger.info("Building Decoder Graph")

        # get positional encodings
        self.keys_positional_encodings = \
            self.get_constant(sinusoidal_position_encoding(self.conf.max_text_sequence_length,
                                                           self.conf.character_embedding_dim,
                                                           position_rate=self.conf.key_position_rate,
                                                           position_weight=1.0))


        self.queries_positional_encodings = \
            self.get_constant(sinusoidal_position_encoding(self.conf.max_spectrogram_length,
                                                           self.conf.decoder_channels,
                                                           position_rate=self.conf.query_position_rate,
                                                           position_weight=1.0))

        with self.namescope("decoder_prenet"):
            x = x_spectrogram
            for dec_pre_ind, dps in enumerate(self.decoder_prenet_sizes):
                if self.dropout_rate > 0.0:
                    x = self.builder.aiOnnx.dropout([x], 1, self.dropout_rate)[0]
                if dec_pre_ind == 0:
                    x = self.apply_speaker_embedding(x, self.mel_bands,
                                                     speaker_embedding, self.speaker_embedding_dim,
                                                     "decoder_pre_{}_apply_speaker_embedding".format(dec_pre_ind))
                    x = self.temp_distributed_FC(x, self.mel_bands, dps,
                                                 "decoder_pre_" + str(dec_pre_ind), activation="relu")
                else:
                    x = self.apply_speaker_embedding(x, self.decoder_prenet_sizes[dec_pre_ind-1],
                                                     speaker_embedding, self.speaker_embedding_dim,
                                                     "decoder_pre_{}_apply_speaker_embedding".format(dec_pre_ind))
                    x = self.temp_distributed_FC(x, self.decoder_prenet_sizes[dec_pre_ind-1], dps,
                                                 "decoder_pre_" + str(dec_pre_ind),
                                                 activation="relu")

        # list to store attention scores from each block
        attention_scores_arrays = []

        with self.namescope("decoder_conv_blocks"):
            for block_ind in range(self.num_decoder_conv_blocks):
                block_name = "deconv_conv_block_" + str(block_ind)
                x = self.gated_residual_conv_block(x, self.decoder_channels, block_name,
                                                   speaker_embedding, self.speaker_embedding_dim,
                                                   dropout_rate=self.dropout_rate)

                if self.decoder_attention_flags[block_ind]:
                    attention_block_name = "decoder_conv_block_attention_" + str(block_ind)
                    context_vecs, attention_scores = \
                        self.attention_block(h_k, h_v, x,
                                             self.character_embedding_dim, self.character_embedding_dim,
                                             self.decoder_channels, self.max_text_sequence_length,
                                             self.attention_hidden_size, attention_block_name,
                                             attention_dropout_rate=self.dropout_rate,
                                             keys_positional_encodings=self.keys_positional_encodings,
                                             queries_positional_encodings=self.queries_positional_encodings)
                    x = self.builder.aiOnnx.mul([self.get_constant(np.sqrt(0.5)),
                                                 self.builder.aiOnnx.add([x, context_vecs])])
                    attention_scores_arrays.append(attention_scores)
        hid = x
        with self.namescope("decoder_postnet"):

            out_size = self.mel_bands * self.n_frames_per_pred

            #  gated linear unit
            x = self.temp_distributed_FC(x, self.decoder_channels, 2 * out_size, "decoder_postnet")
            xs1, xs2 = self.builder.aiOnnx.split([x], num_outputs=2, axis=1)
            xs2_gated = self.builder.aiOnnx.sigmoid([xs2])
            x = self.builder.aiOnnx.mul([xs1, xs2_gated])
            x = self.builder.aiOnnx.sigmoid([x])

            done_flags = self.temp_distributed_FC(hid, self.decoder_channels, 1,
                                                  "decoder_done_block", activation='sigmoid')

        return x, attention_scores_arrays, hid, done_flags

    def __build_graph_for_inference(self, h_k, h_v, speaker_embedding):

        raise NotImplementedError("Autoregressive Inference not implemented yet!")


class Converter(Component):
    def __init__(self, conf, builder, graph_initial_weights=None, graph_name_to_tensor_map=None):
        super(Converter, self).__init__(conf, builder, conv_type="same",
                                        graph_initial_weights=graph_initial_weights,
                                        graph_name_to_tensor_map=graph_name_to_tensor_map)
        self.num_converter_conv_blocks = conf.num_converter_conv_blocks
        self.converter_channels = conf.converter_channels
        self.n_fft = conf.n_fft
        self.n_frames_per_pred = conf.n_frames_per_pred
        self.dropout_rate = conf.dropout_rate


    def __call__(self, x, name_to_tensor):
        self.name_to_tensor = name_to_tensor
        return self.__build_graph(x)

    def __build_graph(self, x):

        logger.info("Building Converter Graph")

        with self.namescope("converter"):

            for block_ind in range(self.num_converter_conv_blocks):
                block_name = "converter_conv_block_" + str(block_ind)
                x = self.gated_residual_conv_block(x, self.converter_channels,
                                                   block_name, dropout_rate=self.dropout_rate)

            x = self.temp_distributed_FC(x, self.converter_channels, self.converter_channels,
                                         "converter_post_1", activation="relu")
            x = self.temp_distributed_FC(x, self.converter_channels,
                                         self.n_frames_per_pred * (self.n_fft//2 + 1),
                                         "converter_post_2", activation="sigmoid")

        return x


class PopartDeepVoice(Component):
    def __init__(self, conf, builder, graph_initial_weights=None, graph_name_to_tensor_map=dict(), for_inference=False):
        super(PopartDeepVoice, self).__init__(conf, builder,
                                              graph_initial_weights=graph_initial_weights,
                                              graph_name_to_tensor_map=graph_name_to_tensor_map)

        self.for_inference = for_inference

        self.encoder = Encoder(conf, builder,
                               graph_initial_weights=graph_initial_weights,
                               graph_name_to_tensor_map=graph_name_to_tensor_map)
        self.decoder = Decoder(conf, builder,
                               graph_initial_weights=graph_initial_weights,
                               graph_name_to_tensor_map=graph_name_to_tensor_map,
                               for_inference=for_inference)
        self.converter = Converter(conf, builder,
                                   graph_initial_weights=graph_initial_weights,
                                   graph_name_to_tensor_map=graph_name_to_tensor_map)

    def __call__(self, x_text, x_spectrogram, speaker_id):

        self.name_to_tensor = dict()

        self.speaker_embedding, speaker_embedding_matrix = self.embedding(speaker_id,
                                                                          self.conf.num_speakers,
                                                                          self.conf.speaker_embedding_dim,
                                                                          "speaker_embedding")

        h_k, h_v = self.encoder(x_text, self.speaker_embedding, self.name_to_tensor)

        # to build inference graph, x_spectrogram must be set to None
        # note that the attention scores arrays format is different for training and inference
        decoder_output, attention_scores_arrays, hid, done_flags = self.decoder(h_k, h_v,
                                                                                x_spectrogram,
                                                                                self.speaker_embedding,
                                                                                self.name_to_tensor)

        mag_spec_out = self.converter(hid, self.name_to_tensor)

        main_outputs = {"mel_spec_output": decoder_output,
                        "mag_spec_output": mag_spec_out,
                        "done_flag_output": done_flags}

        aux_outputs = {"attention_scores_arrays": attention_scores_arrays,
                       "speaker_embedding_matrix": speaker_embedding_matrix}

        return main_outputs, aux_outputs, self.name_to_tensor


def sinusoidal_position_encoding(num_positions, num_channels, position_rate=1.0, position_weight=1.0):
    """ Returns a sinusoidal position encoding table """

    position_encoding = np.array([
        [position_rate * pos / np.power(10000, 2 * (i // 2) / num_channels) for i in range(num_channels)]
        if pos != 0 else np.zeros(num_channels) for pos in range(num_positions)])

    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])  # even i
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])  # odd i

    return position_weight * position_encoding.T
