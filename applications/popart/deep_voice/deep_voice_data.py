# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torchaudio
import random
import numpy as np
from functools import partial
import librosa
import threading

import audio_utils
from audio_utils import REF_DB, MIN_DB


class PadSpectrogram(object):
    """ Function for padding spectrograms """
    def __init__(self, max_spec_length):
        self.max_spec_length = max_spec_length

    def __call__(self, spectrogram):
        pad = self.max_spec_length - spectrogram.shape[-1]
        if pad <= 0:
            done_labels = np.zeros((1, self.max_spec_length))
            return spectrogram[:, 0:self.max_spec_length], done_labels
        done_labels = [0] * spectrogram.shape[-1] + [1] * pad
        done_labels = np.array(done_labels).reshape((1, self.max_spec_length))
        return np.pad(spectrogram, ((0, 0), (0, pad)), "constant"), done_labels

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TransformedVCTKDataSet(torchaudio.datasets.VCTK_092):
    def __init__(self, conf, download=True, transform=None):
        self.conf = conf
        if conf.generated_data:
            # using dummy generated data (no need for setup)
            # create dummy data corresponding to 50K samples
            self.num_steps_train_set = int(5e4 // (conf.batch_size * conf.batches_per_step))
            self.num_steps_valid_set = self.num_steps_train_set
            return
        super(TransformedVCTKDataSet, self).__init__(conf.data_dir,
                                                     download=download)

        self.spectrogram_pad_fn = PadSpectrogram(conf.max_spectrogram_length * conf.n_frames_per_pred)

        if conf.downsample:
            # audio down-sampling function
            self.downsample_fn = partial(librosa.core.resample,
                                         orig_sr=48000,
                                         target_sr=conf.sample_rate)

        # function to compute Short-time Fourier transform of audio signal
        self.get_stft = partial(librosa.core.spectrum.stft,
                                n_fft=conf.n_fft,
                                hop_length=conf.hop_length,
                                win_length=conf.win_length)

        # setting up filter-bank matrix to combine FFT bins into mel-frequency bins
        self.mel_basis = librosa.filters.mel(conf.sample_rate,
                                             conf.n_fft,
                                             fmin=conf.fmin, fmax=conf.fmax,
                                             n_mels=conf.mel_bands,
                                             htk=False)

        # function to convert linear-scale spectrogram to mel-scale spectrogram
        self.convert_linear_spec_to_mel_spec = partial(np.dot,
                                                       a=self.mel_basis)

        # assign each speaker name a unique id
        speaker_name_set = set(self._speaker_ids)
        self.speaker_id_dict = dict((speaker_name, ind) for ind, speaker_name in enumerate(speaker_name_set))

        self.proportion_train_set = conf.proportion_train_set
        self.indices = list(range(len(self)))
        # randomize the order of samples for training
        random.seed(conf.data_shuffling_seed)
        random.shuffle(self.indices)

        self.training_set_base = 0
        # number of samples per session step
        self.step_size = conf.batch_size * conf.batches_per_step

        self.samples_in_training_set = int(self.proportion_train_set * len(self))
        self.samples_in_validation_set = len(self) - self.samples_in_training_set

        self.num_steps_train_set = int(self.samples_in_training_set // self.step_size)
        self.num_steps_valid_set = int(self.samples_in_validation_set // self.step_size)

        if self.samples_in_training_set % self.step_size == 0:
            self.validation_set_base = int(self.samples_in_training_set // self.step_size)
        else:
            self.validation_set_base = int(self.samples_in_training_set // self.step_size + 1)

    def __getitem__(self, n):

        waveform, sample_rate, utterance, speaker_name, utterance_id = super(TransformedVCTKDataSet, self).__getitem__(n)
        waveform = waveform.numpy().squeeze()
        # first down-sample if required
        if self.conf.downsample:
            waveform = self.downsample_fn(waveform)
        # trim out silence
        waveform_trimmed, _ = librosa.effects.trim(waveform, top_db=60.0, ref=100.0)
        if len(waveform_trimmed) < 0.5 * self.conf.sample_rate:  # less than half sec
            waveform = waveform
        else:
            waveform = waveform_trimmed
        # do pre-emphasis
        waveform = audio_utils.preemphasis(x=waveform)

        # compute linear-scale spectrogram
        mag_spectrum = np.abs(self.get_stft(waveform))
        log_mag_spectrum = librosa.core.amplitude_to_db(mag_spectrum, ref=REF_DB, top_db=-MIN_DB)
        log_mag_spectrum_normalized = np.clip((log_mag_spectrum - MIN_DB) / (-MIN_DB), 0.0, 1.0)

        # compute mel-scale spectrogram
        mel_spectrogram = self.convert_linear_spec_to_mel_spec(b=mag_spectrum)
        log_mel = librosa.core.amplitude_to_db(mel_spectrogram, ref=REF_DB, top_db=-MIN_DB)
        log_mel_normalized = np.clip((log_mel - MIN_DB) / (-MIN_DB), 0.0, 1.0)

        # speaker-id
        speaker_id = self.speaker_id_dict[speaker_name]

        # do spectrogram paddings and create done-flag-labels
        log_mag_spectrum_normalized, _ = self.spectrogram_pad_fn(log_mag_spectrum_normalized)
        log_mel_normalized, done_labels = self.spectrogram_pad_fn(log_mel_normalized)

        if self.conf.n_frames_per_pred > 1:
            # reshaping mag spectrum array
            shape = log_mag_spectrum_normalized.shape
            new_shape = (shape[0] * self.conf.n_frames_per_pred, int(shape[1] / self.conf.n_frames_per_pred))
            log_mag_spectrum_normalized = np.reshape(log_mag_spectrum_normalized,
                                                     new_shape)
            # reshaping mel spectrum array
            shape = log_mel_normalized.shape
            new_shape = (shape[0] * self.conf.n_frames_per_pred, int(shape[1] / self.conf.n_frames_per_pred))
            log_mel_normalized = np.reshape(log_mel_normalized, new_shape)

            done_labels = done_labels[0, 0::self.conf.n_frames_per_pred]

        return log_mel_normalized, utterance, speaker_id, log_mag_spectrum_normalized, done_labels

    def get_step_data_iterator(self, train_mode=True):
        """ returns iterator that yields data for each session step """

        def batch_reshape(in_array, batch_shape, dtype):
            out = np.array(in_array)
            out = out.reshape(batch_shape + list(out.shape[1:]))
            if dtype is not None:
                out = out.astype(dtype)
            return out

        conf = self.conf
        if conf.generated_data:
            for dummy_step_data in self.get_dummy_step_data_iterator():
                yield dummy_step_data
            return

        # Determine the shape of the step-data based on batch size, batches_per_step and replication factor
        batch_shape = [conf.samples_per_device]
        if conf.replication_factor > 1:
            batch_shape = [conf.replication_factor] + batch_shape

        if conf.batches_per_step > 1:
            batch_shape = [conf.batches_per_step] + batch_shape

        num_samples_per_step = conf.batch_size * conf.batches_per_step
        samples_in_set = self.samples_in_training_set if train_mode else self.samples_in_validation_set
        num_steps_per_epoch = int(samples_in_set // num_samples_per_step)

        idx_base = self.training_set_base if train_mode else self.validation_set_base
        for step_ind in range(num_steps_per_epoch):
            # the step_data list contains in order mel-scale-spectrogram, utterance, speaker_id,
            # linear-scale-spectrogram and done_labels
            item_data_types = [conf.precision, None, 'int32', conf.precision, 'int32']
            step_data = [[], [], [], [], []]
            if conf.not_multi_thread_dataloader:
                for batch_ind in range(conf.batches_per_step):
                    for sample_ind in range(conf.batch_size):
                        abs_sample_ind = step_ind * num_samples_per_step + \
                                         batch_ind * conf.batch_size + \
                                         sample_ind + idx_base
                        abs_sample_ind = self.indices[abs_sample_ind]
                        sample_data = self[abs_sample_ind]

                        mel_spec_sample, text_sample, speaker_id_sample, mag_spec_sample, done_labels_sample = \
                            sample_data

                        step_data[0].append(mel_spec_sample)
                        step_data[1].append(text_sample)
                        step_data[2].append(speaker_id_sample)
                        step_data[3].append(mag_spec_sample)
                        step_data[4].append(done_labels_sample)
            else:
                num_threads = conf.num_threads
                for item_ind in range(len(item_data_types)):
                    step_data[item_ind] = [None] * num_samples_per_step

                lock = threading.Lock()
                th_arg = [num_samples_per_step, num_threads, step_ind]

                def load_sample_data(thread_id, step_d0, step_d1, step_d2, step_d3, step_d4, thread_arg):

                    num_samples_per_step, num_threads, step_ind = thread_arg
                    thread_index = thread_id

                    while thread_index < num_samples_per_step:
                        with lock:
                            if thread_index < num_samples_per_step:
                                thread_abs_sample_ind = thread_index
                                thread_index += num_threads
                            else:
                                break

                        sample_data_idx = step_ind * num_samples_per_step + thread_abs_sample_ind + idx_base
                        sample_data_idx = self.indices[sample_data_idx]
                        sample_data = self[sample_data_idx]

                        mel_spec_sample, text_sample, speaker_id_sample, mag_spec_sample, done_labels_sample = \
                            sample_data

                        step_d0[thread_abs_sample_ind] = mel_spec_sample
                        step_d1[thread_abs_sample_ind] = text_sample
                        step_d2[thread_abs_sample_ind] = speaker_id_sample
                        step_d3[thread_abs_sample_ind] = mag_spec_sample
                        step_d4[thread_abs_sample_ind] = done_labels_sample

                threads = []
                for i in range(num_threads):
                    t = threading.Thread(target=load_sample_data, args=(i, step_data[0], step_data[1],
                                                                        step_data[2], step_data[3], step_data[4],
                                                                        th_arg,))
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

    def get_dummy_step_data_iterator(self):
        """ returns iterator that generates dummy data for each session step """

        conf = self.conf

        # Determine the shape of the step-data based on batch size, batches_per_step and replication factor
        batch_shape = [conf.samples_per_device]
        if conf.replication_factor > 1:
            batch_shape = [conf.replication_factor] + batch_shape

        if conf.batches_per_step > 1:
            batch_shape = [conf.batches_per_step] + batch_shape

        for _ in range(self.num_steps_train_set):

            mel_spec_data = np.random.uniform(-1, 1,
                                              batch_shape +
                                              [conf.mel_bands, conf.max_spectrogram_length]).astype(conf.precision)
            text_data = np.array(['A' * conf.max_text_sequence_length] *
                                 conf.batch_size * conf.batches_per_step).reshape(batch_shape)
            speaker_data = np.zeros(batch_shape).astype(np.int32)
            mag_spec_data = np.random.uniform(-1, 1,
                                              batch_shape +
                                              [(conf.n_fft // 2 + 1), conf.max_spectrogram_length]).astype(conf.precision)
            done_data = np.zeros(batch_shape + [1, conf.max_spectrogram_length]).astype(np.int32)

            yield mel_spec_data, text_data, speaker_data, mag_spec_data, done_data
