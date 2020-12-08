# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import os
import sys
import librosa
import logging_util

import conf_utils
import audio_utils
from audio_utils import REF_DB, MIN_DB
import text_utils
from deep_voice_train import create_inputs_for_training, create_model_and_dataflow_for_training

# set up logging
logger = logging_util.get_basic_logger(__name__)


if __name__ == '__main__':

    logger.info("Deep Voice Synthesis using feed-forward of training graph.\n"
                "This script only simulates the auto-regressive process for inference.\n"
                "True auto-regressive inference will be coming soon!")

    parser = conf_utils.add_conf_args(run_mode='non_autoregressive_synthesis')
    conf = conf_utils.get_conf(parser)
    if conf.replication_factor > 1:
        logger.error("This script does not support replication-factor > 1")
        sys.exit(-1)
    session_options = conf_utils.get_session_options(conf)
    device = conf_utils.get_device(conf)

    builder = popart.Builder()
    # define inputs
    deep_voice_model_inputs = create_inputs_for_training(builder, conf)

    anchor_mode = 'inference'
    proto, loss_dict, dataflow, main_outputs, aux_outputs, _ = \
        create_model_and_dataflow_for_training(builder, conf, deep_voice_model_inputs, anchor_mode=anchor_mode)

    logger.info("Creating an Inference Session for Synthesis")
    inference_session, anchors = conf_utils.create_session_anchors(proto,
                                                                   loss_dict["total_loss"],
                                                                   device,
                                                                   dataflow,
                                                                   session_options,
                                                                   training=False,
                                                                   optimizer=None,
                                                                   profile=conf.profile)

    if not os.path.exists(conf.trained_model_file):
        logger.error("Model file does not exist: {}".format(conf.trained_model_file))
        sys.exit(-1)
    if not os.path.exists(conf.results_path):
        logger.info("Creating results folder: {}".format(conf.results_path))
        os.makedirs(conf.results_path)

    # copy trained weights onto the device
    logger.info("Loading model to IPU")
    inference_session.resetHostWeights(conf.trained_model_file)
    inference_session.weightsFromHost()

    mel_spec_data = np.zeros((conf.batches_per_step, conf.batch_size,
                              conf.mel_bands, conf.max_spectrogram_length)).astype(np.float32)
    mag_spec_data = np.zeros((conf.batches_per_step, conf.batch_size,
                              (conf.n_fft//2 + 1), conf.max_spectrogram_length)).astype(np.float32)
    done_data = np.zeros((conf.batches_per_step, conf.batch_size, 1, conf.max_spectrogram_length)).astype(np.int32)

    speaker_data = np.zeros((conf.batches_per_step, conf.batch_size)).astype(np.int32)
    text_data = np.zeros((conf.batches_per_step, conf.batch_size, conf.max_text_sequence_length)).astype(np.int32)

    utterance = conf.sentence
    utterance_sequence = text_utils.text_to_sequence(utterance, replace_prob=1.0)
    utterance_sequence = np.array(text_utils.pad_text_sequence(utterance_sequence, conf.max_text_sequence_length))

    for batch_ind in range(conf.batches_per_step):
        for sample_ind in range(conf.batch_size):

            speaker_ind = (batch_ind * conf.batch_size + sample_ind) % conf.num_speakers
            speaker_data[batch_ind, sample_ind] = speaker_ind
            text_data[batch_ind, sample_ind, :] = utterance_sequence

    for frame_ind in range(conf.max_spectrogram_length-1):

        stepio = popart.PyStepIO(
            {
                deep_voice_model_inputs["text_input"]: text_data,
                deep_voice_model_inputs["mel_spec_input"]: mel_spec_data,
                deep_voice_model_inputs["speaker_id"]: speaker_data,
                deep_voice_model_inputs["mag_spec_input"]: mag_spec_data,
                deep_voice_model_inputs["done_labels"]: done_data,
            }, anchors)
        inference_session.run(stepio)

        mel_spec_output = anchors[main_outputs["mel_spec_output"]]
        mel_spec_data[:, :, :, frame_ind + 1] = mel_spec_output[:, :, :, frame_ind]

    for batch_ind in range(conf.batches_per_step):
        for sample_ind in range(conf.batch_size):
            mag_spec_output = anchors[main_outputs["mag_spec_output"]][batch_ind][sample_ind]
            S_db = mag_spec_output * (-MIN_DB) + MIN_DB
            S = librosa.core.db_to_amplitude(S_db, ref=REF_DB)
            audio = librosa.griffinlim(S ** conf.power, hop_length=conf.hop_length, win_length=conf.win_length)
            audio = audio_utils.inv_preemphasis(x=audio)
            audio = audio * conf.amp_factor
            out_wave_fp = os.path.join(conf.results_path,
                                       'speaker_{}_generated.wav'.format(speaker_data[batch_ind, sample_ind]))
            librosa.output.write_wav(out_wave_fp, audio, conf.sample_rate)
            logger.info('Output audio synthesized at: {}'.format(out_wave_fp))

            mag_spec_out_file = os.path.join(conf.results_path,
                                             'mag_spec_output_{}.npz'.format(speaker_data[batch_ind, sample_ind]))
            np.savez(mag_spec_out_file, mag_spec_output=mag_spec_output)

            mel_spec_output = anchors[main_outputs["mel_spec_output"]][batch_ind][sample_ind]

            mel_spec_out_file = os.path.join(conf.results_path,
                                             'mel_spec_output_{}.npz'.format(speaker_data[batch_ind, sample_ind]))
            np.savez(mel_spec_out_file, mel_spec_output=mel_spec_output)


    for batch_ind in range(conf.batches_per_step):
        for sample_ind in range(conf.batch_size):

            for att_ind, attention_distribution in enumerate(aux_outputs["attention_scores_arrays"]):
                attention_scores = anchors[attention_distribution][batch_ind][sample_ind]

                att_fp = os.path.join(conf.results_path,
                                      'att_{}_attind_{}.npz'.format(speaker_data[batch_ind, sample_ind],
                                                                    att_ind))
                np.savez(att_fp, attention_scores=attention_scores)
                logger.info("Saved Attention Results to {}".format(att_fp))
