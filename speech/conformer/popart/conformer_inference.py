# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import os
import torch

import logging_util
import conf_utils
import text_utils
import librispeech_data
import conformer_builder
import ctcdecode

# set up logging
logger = logging_util.get_basic_logger('CONFORMER_INFERENCE')


def _get_popart_type(np_type):
    return {
        np.float16: 'FLOAT16',
        np.float32: 'FLOAT'
    }[np_type]


def create_inputs_for_inference(builder, conf):
    """ defines the input tensors for the conformer model """

    inputs = dict()

    inputs["mel_spec_input"] = builder.addInputTensor(popart.TensorInfo(_get_popart_type(conf.precision),
                                                                        [conf.samples_per_device,
                                                                         conf.mel_bands,
                                                                         conf.max_spectrogram_length]),
                                                      "mel_spec_input")

    return inputs


def create_model_and_dataflow_for_inference(builder, conf, inputs):
    """ builds the conformer model and dataflow for inference """

    conformer_encoder = conformer_builder.ConformerEncoder(builder,
                                                           input_dim=conf.mel_bands,
                                                           sequence_length=conf.max_spectrogram_length,
                                                           encoder_dim=conf.encoder_dim,
                                                           attention_heads=conf.attention_heads,
                                                           encoder_layers_per_stage=conf.encoder_layers_per_stage,
                                                           dropout_rate=conf.dropout_rate,
                                                           cnn_module_kernel=conf.kernel_size,
                                                           subsampling_factor=conf.subsampling_factor,
                                                           dtype=conf.precision)

    conformer_decoder = conformer_builder.ConformerDecoder(builder,
                                                           encoder_dim=conf.encoder_dim,
                                                           num_symbols=conf.num_symbols,
                                                           for_inference=True,
                                                           dtype=conf.precision)

    encoder_output = conformer_encoder(inputs["mel_spec_input"])

    with builder.virtualGraph(conf.num_pipeline_stages - 1):
        probs_output = conformer_decoder(encoder_output)

    anchor_types_dict = {
        probs_output: popart.AnchorReturnType("ALL"),
    }

    proto = builder.getModelProto()
    dataflow = popart.DataFlow(conf.device_iterations, anchor_types_dict)

    return proto, probs_output, dataflow


if __name__ == '__main__':

    logger.info("Conformer Inference in Popart")

    parser = conf_utils.add_conf_args(run_mode='inference')
    conf = conf_utils.get_conf(parser)
    session_options = conf_utils.get_session_options(conf)
    device = conf_utils.get_device(conf)

    if not os.path.exists(conf.results_dir):
        logger.info("Creating results directory {}".format(conf.results_dir))
        os.makedirs(conf.results_dir)
    results_filepath = os.path.join(conf.results_dir, 'inference_results.txt')
    open(results_filepath, 'w').close()

    # building model and dataflow
    builder = popart.Builder()
    conformer_model_inputs = create_inputs_for_inference(builder, conf)

    proto, probs_output, dataflow = create_model_and_dataflow_for_inference(builder,
                                                                            conf,
                                                                            conformer_model_inputs)

    # create inference session
    logger.info("Creating the inference session")
    inference_session, anchors = \
        conf_utils.create_session_anchors(proto,
                                          [],
                                          device,
                                          dataflow,
                                          session_options,
                                          training=False,
                                          optimizer=None)
    # copy trained weights onto the device
    logger.info("Loading model to IPU")
    inference_session.resetHostWeights(conf.model_file)
    inference_session.weightsFromHost()

    logger.info("Preparing LibriSpeech dataset for testing")
    dataset = librispeech_data.LibriSpeechDataset(conf)
    logger.info("Number of clips in {} for testing: {}".format(conf.dataset, len(dataset)))

    dataset_iterator = dataset.get_step_data_iterator()

    ctc_beam_decoder = ctcdecode.CTCBeamDecoder(text_utils.symbols, beam_width=20,
                                                blank_id=text_utils.symbols.index(text_utils._blank_symbol_ctc))

    for mel_spec_data, text_data, ctc_input_length_data, ctc_target_length_data in dataset_iterator:
        stepio = popart.PyStepIO(
            {
                conformer_model_inputs["mel_spec_input"]: mel_spec_data,
            }, anchors)

        inference_session.run(stepio)

        # collects all output probability data from inference session run
        probs_output_data = anchors[probs_output]

        for step_ind in range(conf.device_iterations):
            for batch_ind in range(conf.batch_size):
                sample_id = step_ind * conf.batch_size + batch_ind

                seq_length = ctc_input_length_data[step_ind, batch_ind]
                probs_seq = torch.FloatTensor(probs_output_data[step_ind, batch_ind, 0:seq_length, :])
                probs_seq = torch.reshape(probs_seq, (1, probs_seq.shape[-2], probs_seq.shape[-1]))

                beam_result, beam_scores, timesteps, out_seq_len = ctc_beam_decoder.decode(probs_seq)

                actual_string = "Actual: {}".format(text_utils.sequence_to_text(text_data[step_ind, batch_ind],
                                                                                conf.max_text_sequence_length))
                decoding_string = "Decoding: {}".format(text_utils.sequence_to_text(beam_result[0][0].tolist(),
                                                                                    out_seq_len[0][0]))

                print(actual_string)
                print(decoding_string)

                with open(results_filepath, 'a') as f:
                    f.write(actual_string + '\n')
                    f.write(decoding_string + '\n\n')
