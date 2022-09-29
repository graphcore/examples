# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torchaudio
import random
import librosa
import threading
import numpy as np
from tqdm import tqdm
import sys

import audio_utils
import text_utils


class LibriSpeechDataset(torchaudio.datasets.LIBRISPEECH):
    """ Creates the LibriSpeech dataset for training of the conformer model """
    def __init__(self, conf, download=True):
        self.conf = conf
        super(LibriSpeechDataset, self).__init__(conf.data_dir,
                                                 url=conf.dataset,
                                                 download=download)
        self.spectrogram_computer = audio_utils.SpectrogramComputer(conf)
        self.ctc_max_input_length = int(conf.max_spectrogram_length / conf.subsampling_factor)
        self.indices = list(range(len(self)))
        # randomize the order of samples for training
        random.seed(1222)
        random.shuffle(self.indices)
        self.step_size = conf.global_batch_size * conf.device_iterations
        self.num_steps = int(len(self) / self.step_size)

    def __getitem__(self, n):

        waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = super(LibriSpeechDataset, self).__getitem__(n)
        waveform = waveform.numpy().squeeze()
        mel_scale_spectrogram = self.spectrogram_computer.compute_mel_scale_spectrogram(waveform,
                                                                                        in_decibels=True)
        utterance_sequence = text_utils.text_to_sequence(utterance)

        # length of target sequence for CTC loss
        ctc_target_length = min(len(utterance_sequence), self.conf.max_text_sequence_length)

        # length of input sequence length for CTC loss (original length / subsampling_factor)
        ctc_input_length = int(mel_scale_spectrogram.shape[1] / self.conf.subsampling_factor)
        ctc_input_length = min(ctc_input_length, self.ctc_max_input_length)

        # do necessary paddings
        mel_scale_spectrogram = audio_utils.pad_spectrogram(mel_scale_spectrogram, self.conf.max_spectrogram_length)
        utterance_sequence = text_utils.pad_text_sequence(utterance_sequence, self.conf.max_text_sequence_length)

        return mel_scale_spectrogram, utterance_sequence, ctc_input_length, ctc_target_length

    def get_step_data_iterator(self):
        """ returns iterator that yields data for each session step """

        def batch_reshape(in_array, batch_shape, dtype):
            out = np.array(in_array)
            out = out.reshape(batch_shape + list(out.shape[1:]))
            if dtype is not None:
                out = out.astype(dtype)
            return out

        conf = self.conf
        # Determine the shape of the step-data based on replica_batch_size, device_iterations and replication_factor
        batch_shape = [conf.replica_batch_size]
        if conf.replication_factor > 1:
            batch_shape = [conf.replication_factor] + batch_shape

        if conf.device_iterations > 1:
            batch_shape = [conf.device_iterations] + batch_shape

        num_samples_per_step = conf.global_batch_size * conf.device_iterations
        samples_in_set = len(self)
        num_steps_per_epoch = int(samples_in_set // num_samples_per_step)

        for step_ind in range(num_steps_per_epoch):
            # the step_data list contains in order mel-scale-spectrogram, utterance-sequence, input_length, target_length
            item_data_types = [conf.precision, 'uint32', 'uint32', 'uint32']
            step_data = [[], [], [], []]
            if conf.not_multi_thread_dataloader:
                for batch_ind in range(conf.device_iterations):
                    for sample_ind in range(conf.global_batch_size):
                        abs_sample_ind = step_ind * num_samples_per_step + \
                                         batch_ind * conf.global_batch_size + \
                                         sample_ind
                        abs_sample_ind = self.indices[abs_sample_ind]
                        sample_data = self[abs_sample_ind]

                        mel_spec_sample, text_sample,  input_length_sample, target_length_sample = sample_data

                        step_data[0].append(mel_spec_sample)
                        step_data[1].append(text_sample)
                        step_data[2].append(input_length_sample)
                        step_data[3].append(target_length_sample)
            else:
                num_threads = conf.num_threads
                for item_ind in range(len(item_data_types)):
                    step_data[item_ind] = [None] * num_samples_per_step

                lock = threading.Lock()
                th_arg = [num_samples_per_step, num_threads, step_ind]

                def load_sample_data(thread_id, step_d0, step_d1, step_d2, step_d3, thread_arg):

                    num_samples_per_step, num_threads, step_ind = thread_arg
                    thread_index = thread_id

                    while thread_index < num_samples_per_step:
                        with lock:
                            if thread_index < num_samples_per_step:
                                thread_abs_sample_ind = thread_index
                                thread_index += num_threads
                            else:
                                break

                        sample_data_idx = step_ind * num_samples_per_step + thread_abs_sample_ind
                        sample_data_idx = self.indices[sample_data_idx]
                        sample_data = self[sample_data_idx]

                        mel_spec_sample, text_sample,  input_length_sample, target_length_sample = sample_data

                        step_d0[thread_abs_sample_ind] = mel_spec_sample
                        step_d1[thread_abs_sample_ind] = text_sample
                        step_d2[thread_abs_sample_ind] = input_length_sample
                        step_d3[thread_abs_sample_ind] = target_length_sample

                threads = []
                for i in range(num_threads):
                    t = threading.Thread(target=load_sample_data,
                                         args=(i, step_data[0], step_data[1], step_data[2], step_data[3], th_arg,))
                    threads.append(t)
                # fire all threads up
                for t in threads:
                    t.start()
                # wait for all threads
                for t in threads:
                    t.join()

            # reshaping step_data for PyStepIO
            for item_ind, item_data_type in enumerate(item_data_types):
                step_data[item_ind] = batch_reshape(step_data[item_ind], batch_shape, item_data_type)

            yield step_data

    def load_all_step_data(self):

        all_step_data = []

        dataset_iterator = self.get_step_data_iterator()
        tqdm_iter = tqdm(dataset_iterator, disable=not sys.stdout.isatty())

        for step_data in tqdm_iter:
            all_step_data.append(step_data)

        return all_step_data
