# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np

import logging_util
import conformer_blocks

# set up logging
logger = logging_util.get_basic_logger(__name__)


class ConformerEncoder(conformer_blocks.Block):
    """ Conformer encoder module.
    :param popart builder: popart builder object
    :param int input_dim: input dimension
    :param int sequence_length: length of sequence
    :param int encoder_dim: encoder dimension
    :param int attention_heads: the number of heads of multi head attention
    :param list encoder_layers_per_stage: list with number of encoder layers per each pipeline stage
    :param float dropout_rate: dropout rate
    :param bool use_conv_module: whether to use convolution module
    :param int cnn_module_kernel: kernel size of convolution module
    """

    def __init__(
            self,
            builder,
            input_dim,
            sequence_length,
            encoder_dim,
            attention_heads=4,
            encoder_layers_per_stage=[2],
            dropout_rate=0.1,
            use_conv_module=True,
            cnn_module_kernel=31,
            subsampling_factor=4,
            dtype=np.float32):

        super(ConformerEncoder, self).__init__(builder, dtype)
        self.dropout_rate = dropout_rate
        self.conv_subsampler = conformer_blocks.ConvolutionSubSampler(builder,
                                                                      input_dim,
                                                                      kernel_size=cnn_module_kernel,
                                                                      subsampling_factor=subsampling_factor,
                                                                      dtype=dtype)
        self.prenet_linear = conformer_blocks.Linear(builder, input_dim, encoder_dim, dtype=dtype)
        self.pipeline_stages = [None] * len(encoder_layers_per_stage)
        for stage_ind, num_enc_layers in enumerate(encoder_layers_per_stage):
            self.pipeline_stages[stage_ind] = [conformer_blocks.ConformerBlock(builder,
                                                                               attention_heads,
                                                                               encoder_dim,
                                                                               sequence_length,
                                                                               kernel_size=cnn_module_kernel,
                                                                               use_conv_module=use_conv_module,
                                                                               dropout_rate=dropout_rate,
                                                                               dtype=dtype)
                                               for _ in range(num_enc_layers)]

    def __call__(self, x_in):
        return self.__build_graph(x_in)

    def __build_graph(self, x_in):

        x = x_in
        logger.info("Shape of Encoder Input: {}".format(self.builder.getTensorShape(x)))
        # place conv-subsampler and prenet-block in first stage
        with self.builder.virtualGraph(0):
            # apply prenet with conv-subsampling
            x = self.conv_subsampler(x)
            x = self.prenet_linear(x)
            x = self.builder.aiOnnx.dropout([x], 1, self.dropout_rate)[0]
            logger.info("Shape after Encoder Prenet: {}".format(self.builder.getTensorShape(x)))
        for stage_ind, stage_blocks in enumerate(self.pipeline_stages):
            with self.builder.virtualGraph(stage_ind):
                # apply conformer blocks
                for block_ind, cb in enumerate(stage_blocks):
                    x = cb(x)
                    logger.info("Shape after Pipeline Stage {}, "
                                "Conformer block {}: {}".format(stage_ind,
                                                                block_ind,
                                                                self.builder.getTensorShape(x)))
        return x


class ConformerDecoder(conformer_blocks.Block):
    """ Conformer decoder module
    (linear projection followed by softmax layer giving probabilities over set of symbols).
    :param int encoder_dim: encoder dimension
    :param int num_symbols: number of text symbols used
    """

    def __init__(
            self,
            builder,
            encoder_dim,
            num_symbols,
            dtype=np.float32,
            for_inference=False):

        super(ConformerDecoder, self).__init__(builder, dtype)

        self.encoder_dim = encoder_dim
        self.num_symbols = num_symbols
        self.for_inference = for_inference

    def __call__(self, x_in):
        return self.__build_graph(x_in)

    def __build_graph(self, x_in):

        logger.info("Shape of Decoder Input: {}".format(self.builder.getTensorShape(x_in)))

        # transposing to shape [global_batch_size, seq_length, channel_dim]
        out = self.builder.aiOnnx.transpose([x_in], perm=[0, 2, 1])

        # matmul for softmax layer
        wshape = [self.encoder_dim, self.num_symbols]
        softmax_init_weights = self.xavier_init(wshape, self.encoder_dim, self.num_symbols)
        softmax_mat = self.add_tensor("softmax_layer_weights", softmax_init_weights)
        logits = self.builder.aiOnnx.matmul([out, softmax_mat])
        if not self.for_inference:
            log_probs = self.builder.aiOnnx.logsoftmax([logits], axis=2)
            logger.info("Shape of Log-Probs output from Decoder: {}".format(self.builder.getTensorShape(log_probs)))
            return log_probs
        else:
            probs = self.builder.aiOnnx.softmax([logits], axis=2)
            logger.info("Shape of Probs output from Decoder (for inference): {}".format(self.builder.getTensorShape(probs)))
            return probs
