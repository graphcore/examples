# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np

import logging_util
import transducer_blocks

# set up logging
logger = logging_util.get_basic_logger(__name__)


class TranscriptionNetwork(transducer_blocks.Block):
    """ Transcription Network (or Audio Encoding network) of the Transformer-Transducer model.
    :param popart builder: popart builder object
    :param int in_feats: input dimension
    :param int subsampling_factor: sub-sampling factor for the initial convolutional subsampling layer
    :param int enc_n_hid: encoder hidden dimension
    :param int num_encoder_layers: the number of transformer layers for the transcription encoding network
    :param int encoder_dim: dimension of the transformer layers
    :param int num_attention_heads: number of attention heads
    :param float enc_dropout: dropout rate for encoder net
    :param int kernel_size: kernel size of the initial convolutional subsampling layer
    """

    def __init__(
            self,
            builder,
            in_feats,
            subsampling_factor,
            num_encoder_layers,
            encoder_dim,
            num_attention_heads,
            enc_dropout,
            kernel_size=32,
            dtype=np.float32):

        super(TranscriptionNetwork, self).__init__(builder, dtype, block_name="transcription_network")
        self.encoder_dim = encoder_dim
        self.subsampling_factor = subsampling_factor

        self.conv_subsampler = transducer_blocks.ConvolutionSubSampler(builder,
                                                                       in_feats,
                                                                       encoder_dim,
                                                                       kernel_size,
                                                                       subsampling_factor,
                                                                       dtype=dtype,
                                                                       block_name="transcription_net_convolution_subsampler")

        self.transformer_blocks = [transducer_blocks.TransformerBlock(builder,
                                                                      num_attention_heads,
                                                                      encoder_dim,
                                                                      dtype=dtype,
                                                                      block_name="transcription_net_transformer_block_{}".format(
                                                                          layer_ind),
                                                                      dropout_rate=enc_dropout)
                                   for layer_ind in range(num_encoder_layers)]

        self.child_blocks = [self.conv_subsampler] + self.transformer_blocks

    def __call__(self, x_in, x_lens):
        return self.__build_graph(x_in, x_lens)

    def __build_graph(self, x_in, x_lens):
        # input shape to transcription-network must be [batch_size, channel_dim, seq_length]

        builder = self.builder
        logger.info("Shape of Transcription-Network Input: {}".format(builder.getTensorShape(x_in)))

        with self.builder.virtualGraph(0):
            x = x_in

            x = self.conv_subsampler(x)
            # scale x_lens as well after subsampling
            x_lens = self.builder.aiOnnx.div([x_lens,
                                              self.builder.aiOnnx.constant(np.array([self.subsampling_factor]).astype('int32'))])

            builder.recomputeOutputInBackwardPass(x)

            # add positional encoding
            seq_length = self.builder.getTensorShape(x)[2]
            positional_encodings = self.get_constant(sinusoidal_position_encoding(seq_length,
                                                                                  self.encoder_dim))
            x = self.builder.aiOnnx.add([x, positional_encodings])

            builder.recomputeOutputInBackwardPass(x)

            for layer_ind, transformer_block in enumerate(self.transformer_blocks):
                x = transformer_block(x, force_recompute=True)

            # transpose to shape [batch_size, seq_length, channel_dim]
            x = builder.aiOnnx.transpose([x], perm=[0, 2, 1])

        return x, x_lens


class PredictionNetwork(transducer_blocks.Block):
    """ Prediction Network of the Transducer model.
    :param popart builder: popart builder object
    :param int num_symbols: number of symbols to embed
    :param int pred_n_hid: hidden dimension for LSTM layers of prediction network
    :param int pred_rnn_layers: number of LSTM layers of prediction network
    :param float pred_dropout: dropout rate for prediction net
    :param float forget_gate_bias: value to initialize the forget gate bias values to
    :param float weights_init_scale: scaling factor for initial weights and biases of LSTM layers
    """

    def __init__(
            self,
            builder,
            num_symbols,
            pred_n_hid,
            pred_rnn_layers,
            pred_dropout,
            forget_gate_bias,
            weights_init_scale,
            dtype=np.float32):

        super(PredictionNetwork, self).__init__(builder, dtype, block_name="prediction_network")
        self.num_symbols = num_symbols
        self.pred_n_hid = pred_n_hid
        self.pred_rnn_layers = pred_rnn_layers
        self.pred_dropout = pred_dropout

        self.embedding_layer = transducer_blocks.EmbeddingBlock(builder,
                                                                num_symbols,
                                                                pred_n_hid,
                                                                dtype=dtype,
                                                                block_name="prediction_net_embedding")

        self.prediction_rnn_layers = [transducer_blocks.LSTM(builder,
                                                             pred_n_hid,
                                                             pred_n_hid,
                                                             dtype=dtype,
                                                             block_name="prediction_net_rnn_{}".format(pred_rnn_ind),
                                                             forget_gate_bias=forget_gate_bias,
                                                             weights_init_scale=weights_init_scale)
                                      for pred_rnn_ind in range(pred_rnn_layers)]

        self.child_blocks = [self.embedding_layer] + self.prediction_rnn_layers

    def __call__(self, x_in):
        return self.__build_graph(x_in)

    def __build_graph(self, x_in):
        # input shape to this layer is assumed to be [batch_size, target_sequence_length]

        builder = self.builder
        logger.info("Shape of Prediction-Network Input: {}".format(builder.getTensorShape(x_in)))

        with self.builder.virtualGraph(0):
            x = x_in
            x = self.embedding_layer(x)
            # input shape to lstm layers must be [seq_length, batch_size, channel_dim]
            x = builder.aiOnnx.transpose([x], perm=[1, 0, 2])

            # prepend blank symbol (zero-vector) to beginning of sequence
            blank_shape = builder.getTensorShape(x)
            blank_shape[0] = 1
            blank_prepend = self.get_constant(np.zeros(blank_shape))
            x = builder.aiOnnx.concat([blank_prepend, x], axis=0)

            for layer_ind, lstm_layer in enumerate(self.prediction_rnn_layers):
                x = lstm_layer(x, force_recompute=True)
                x = self.apply_dropout(x, self.pred_dropout)
                logger.info("Shape after Pred-RNN layer {}: {}".format(layer_ind, builder.getTensorShape(x)))

            # transposing back to shape [batch_size, seq_length, channel_dim]
            x = builder.aiOnnx.transpose([x], perm=[1, 0, 2])

        return x


class JointNetwork(transducer_blocks.Block):
    """ Joint Network of the Transducer model.
    :param popart builder: popart builder object
    :param int enc_n_hid: encoder hidden dimension
    :param int pred_n_hid: hidden dimension for LSTM layers of prediction network
    :param int joint_n_hid: hidden dimension of Joint Network
    :param int num_symbols: number of symbols to embed
    :param float joint_dropout: dropout rate for joint net
    """

    def __init__(
            self,
            builder,
            transcription_out_len,
            enc_n_hid,
            pred_n_hid,
            joint_n_hid,
            num_symbols,
            joint_dropout,
            dtype=np.float32,
            transcription_out_split_size=15,
            shift_labels_by_one=True):

        super(JointNetwork, self).__init__(builder, dtype, block_name="joint_network")
        self.enc_n_hid = enc_n_hid
        self.pred_n_hid = pred_n_hid
        self.joint_n_hid = joint_n_hid
        self.num_symbols = num_symbols
        self.joint_dropout = joint_dropout

        logger.warn("For best training performance it is recommended that "
                    "transcription output split size({}) be a divisor of "
                    "transcription output length({}).".format(transcription_out_split_size, transcription_out_len))

        self.joint_transcription_fc = transducer_blocks.RHSLinear(builder,
                                                                  enc_n_hid,
                                                                  joint_n_hid,
                                                                  dtype=dtype,
                                                                  block_name="joint_net_transcription_fc")
        self.joint_prediction_fc = transducer_blocks.RHSLinear(builder,
                                                               pred_n_hid,
                                                               joint_n_hid,
                                                               dtype=dtype,
                                                               block_name="joint_net_prediction_fc")
        self.transcription_splitter = transducer_blocks.Split(builder,
                                                              total_size=transcription_out_len,
                                                              split_size=transcription_out_split_size,
                                                              split_axis=1,
                                                              dtype=dtype,
                                                              block_name="joint_net_transcription_splitter")
        self.joint_out_fc = transducer_blocks.RHSLinear(builder,
                                                        joint_n_hid,
                                                        num_symbols,
                                                        dtype=dtype,
                                                        block_name='joint_net_out_fc')

        self.child_blocks = [self.joint_transcription_fc, self.joint_prediction_fc,
                             self.transcription_splitter, self.joint_out_fc]

        self.shift_labels_by_one = shift_labels_by_one

    def __call__(self, transcription_out, transcription_lens, prediction_out, targets, target_lens):
        return self.__build_graph(transcription_out, transcription_lens, prediction_out, targets, target_lens)

    def get_log_probs(self, transcription_out_split, prediction_out, targets, target_lens):

        builder = self.builder

        joint_out_split = builder.aiOnnx.add([transcription_out_split, prediction_out])
        builder.recomputeOutputInBackwardPass(joint_out_split)
        joint_out_split = builder.aiOnnx.relu([joint_out_split])
        builder.recomputeOutputInBackwardPass(joint_out_split)
        joint_out_split = self.apply_dropout(joint_out_split, self.joint_dropout)
        builder.recomputeOutputInBackwardPass(joint_out_split)

        joint_out_split = self.joint_out_fc(joint_out_split, force_recompute=True)

        # This flag means we need to offset labels by + 1 when passing to RNN-T Loss
        # The reason for offset is that we treat logits "A" dimension as [<blank>, valid characters... A-1]
        # Thus, blank-symbol has idx 0 and real symbols must have indices [1:A-1]
        # RNN-T Loss uses labels as indices of logits (in A dimension)
        # The opposite logic must be applied when logits are used for decoder - see transducer_decoder.py
        if self.shift_labels_by_one:
            one = self.builder.aiOnnx.constant(np.array([1]).astype(np.int32))
            targets = self.builder.aiOnnx.add([targets, one])

        compact_log_probs = builder.customOp(
            opName="SparseLogSoftmax",
            opVersion=1,
            domain="com.acme",
            inputs=[joint_out_split, targets, target_lens],
            attributes={},
            numOutputs=1,
        )
        return compact_log_probs[0]

    def __build_graph(self, transcription_out, transcription_lens, prediction_out, targets, target_lens):

        builder = self.builder
        logger.info("Shapes of Joint-Network Inputs: {}, {}".format(builder.getTensorShape(transcription_out),
                                                                    builder.getTensorShape(prediction_out)))

        with self.builder.virtualGraph(0):
            transcription_out = self.joint_transcription_fc(transcription_out, force_recompute=True)
            prediction_out = self.joint_prediction_fc(prediction_out, force_recompute=True)

            transcription_out = self.builder.aiOnnx.unsqueeze([transcription_out], axes=[2])
            prediction_out = self.builder.aiOnnx.unsqueeze([prediction_out], axes=[1])

            transcription_out_splits = self.transcription_splitter(transcription_out)

        log_probs_compact_splits = []
        for split_ind, transcription_out_split in enumerate(transcription_out_splits):
            logger.info("Building compact log probs for split {}".format(split_ind))
            with self.builder.virtualGraph(0):
                log_probs_compact = self.get_log_probs(transcription_out_split, prediction_out,
                                                       targets, target_lens)
                log_probs_compact_splits.append(log_probs_compact)

        with self.builder.virtualGraph(0):
            # stack all compacted logprobs
            log_probs_compact_out = builder.aiOnnx.concat(log_probs_compact_splits, axis=1)

        # logger.info("Shape of Joint-Network output: {}".format(builder.getTensorShape(log_probs_compact_out)))

        return log_probs_compact_out


class RNNTLoss(transducer_blocks.Block):
    """ Returns RNN-T loss value for given inputs """
    def __init__(self, builder, dtype):
        super(RNNTLoss, self).__init__(builder, dtype, block_name="RNNTLoss")

    def __call__(self, joint_out, input_length, target_length):
        return self.__build_graph(joint_out, input_length, target_length)

    def __build_graph(self, joint_out, input_length, target_length):

        with self.namescope("RNNTLoss"):

            builder = self.builder

            with builder.virtualGraph(0):
                rnnt_outputs = builder.customOp(
                    opName="RNNTLoss",
                    opVersion=1,
                    domain="com.acme",
                    inputs=[joint_out, input_length, target_length],
                    attributes={},
                    numOutputs=4,
                )
                neg_log_likelihood = rnnt_outputs[0]

        return neg_log_likelihood


class JointNetwork_wRNNTLoss(transducer_blocks.Block):
    """ Joint Network of the RNN-T model followed by RNN-Transducer loss.
    :param popart builder: popart builder object
    :param int transcription_out_len: sequence length of the transcription net output
    :param int enc_n_hid: encoder hidden dimension
    :param int pred_n_hid: hidden dimension for LSTM layers of prediction network
    :param int joint_n_hid: hidden dimension of Joint Network
    :param int num_symbols: number of symbols to embed
    :param float joint_dropout: dropout rate for joint net
    """

    def __init__(
            self,
            builder,
            transcription_out_len,
            enc_n_hid,
            pred_n_hid,
            joint_n_hid,
            num_symbols,
            joint_dropout,
            dtype=np.float32,
            transcription_out_split_size=15,
            do_batch_serialization=False,
            samples_per_device=2,
            batch_split_size=1,
            shift_labels_by_one=True):
        super(JointNetwork_wRNNTLoss, self).__init__(builder, dtype, block_name="joint_network_w_rnnt_loss")

        self.joint_network = JointNetwork(builder,
                                          transcription_out_len,
                                          enc_n_hid,
                                          pred_n_hid,
                                          joint_n_hid,
                                          num_symbols,
                                          joint_dropout,
                                          dtype=dtype,
                                          transcription_out_split_size=transcription_out_split_size,
                                          shift_labels_by_one=shift_labels_by_one)

        self.rnnt_loss = RNNTLoss(builder, dtype=dtype)
        self.do_batch_serialization = do_batch_serialization
        self.samples_per_device = samples_per_device
        self.batch_split_size = batch_split_size

        self.child_blocks = [self.joint_network, self.rnnt_loss]

        if do_batch_serialization:
            self.batch_splitter = transducer_blocks.Split(builder, total_size=samples_per_device,
                                                          split_size=batch_split_size, split_axis=0,
                                                          dtype=dtype, block_name="batch_splitter")
            self.child_blocks.append(self.batch_splitter)


    def __call__(self, transcription_out, transcription_lens, prediction_out, targets, target_lens):
        return self.__build_graph(transcription_out, transcription_lens, prediction_out, targets, target_lens)

    def __build_graph(self, transcription_out, transcription_lens, prediction_out, targets, target_lens):

        builder = self.builder

        if not self.do_batch_serialization:
            joint_out = self.joint_network(transcription_out, transcription_lens, prediction_out,
                                           targets, target_lens)

            neg_log_likelihood = self.rnnt_loss(joint_out, transcription_lens, target_lens)
            return neg_log_likelihood
        else:
            logger.info("Doing Batch-Serialization for JointNet")
            tout_splits = self.batch_splitter(transcription_out)
            flen_splits = self.batch_splitter(transcription_lens)
            pout_splits = self.batch_splitter(prediction_out)
            tin_splits = self.batch_splitter(targets)
            tlen_splits = self.batch_splitter(target_lens)

            losses = []
            for tout, flen, pout, tin, tlen in zip(tout_splits, flen_splits, pout_splits, tin_splits, tlen_splits):
                joint_out = self.joint_network(tout, flen, pout, tin, tlen)
                losses.append(self.rnnt_loss(joint_out, flen, tlen))
            reduced_neg_log_likelihood = losses[0]
            for loss in losses[1:]:
                reduced_neg_log_likelihood = builder.aiOnnx.add([reduced_neg_log_likelihood, loss])
            reduced_neg_log_likelihood = builder.aiOnnx.div([reduced_neg_log_likelihood,
                                                             builder.aiOnnx.constant(
                                                                 np.array([len(losses)]).astype(np.float32))])
            return reduced_neg_log_likelihood


def sinusoidal_position_encoding(num_positions, num_channels, position_rate=1.0, position_weight=1.0):
    """ Returns a sinusoidal position encoding table """

    position_encoding = np.array([
        [position_rate * pos / np.power(10000, 2 * (i // 2) / num_channels) for i in range(num_channels)]
        if pos != 0 else np.zeros(num_channels) for pos in range(num_positions)])

    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])  # even i
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])  # odd i

    return position_weight * position_encoding.T
