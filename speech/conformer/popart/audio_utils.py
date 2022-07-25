# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
from functools import partial
import librosa

REF_DB = 20.0  # reference used for scaling spectrogram
MIN_DB = -100.0  # min decibel value used for clipping spectrograms


class SpectrogramComputer(object):
    """ Provides functions for computing spectograms for given configuration parameters """
    def __init__(self, conf):

        hop_length_samples = int(conf.hop_length_ms * 1e-3 * conf.sample_rate)
        win_length_samples = int(conf.win_length_ms * 1e-3 * conf.sample_rate)

        # function to compute Short-time Fourier transform of audio signal
        self.get_stft = partial(librosa.core.spectrum.stft,
                                n_fft=win_length_samples,
                                hop_length=hop_length_samples,
                                win_length=win_length_samples)

        # setting up filter-bank matrix to combine FFT bins into mel-frequency bins
        self.mel_basis = librosa.filters.mel(conf.sample_rate,
                                             win_length_samples,
                                             fmin=conf.fmin, fmax=conf.fmax,
                                             n_mels=conf.mel_bands,
                                             htk=False)

    def compute_linear_scale_spectrogram(self, waveform):
        """ Computes linear scale spectrogram for given waveform """
        linear_scale_spectrogram = np.abs(self.get_stft(waveform))
        return linear_scale_spectrogram

    def compute_mel_scale_spectrogram(self, waveform, in_decibels=True):
        """ Computes mel-scale spectrogram for given waveform """
        linear_scale_spectrogram = self.compute_linear_scale_spectrogram(waveform)
        mel_scale_spectrogram = np.dot(a=self.mel_basis, b=linear_scale_spectrogram)
        if in_decibels:
            mel_scale_spectrogram = librosa.core.amplitude_to_db(mel_scale_spectrogram,
                                                                 ref=REF_DB, top_db=-MIN_DB)
            mel_scale_spectrogram = np.clip((mel_scale_spectrogram - MIN_DB) / (-MIN_DB), 0.0, 1.0)
        return mel_scale_spectrogram


def pad_spectrogram(spectrogram, max_spec_length):
    """ Function for padding spectrograms """
    pad = max_spec_length - spectrogram.shape[-1]
    if pad <= 0:
        return spectrogram[:, 0:max_spec_length]
    return np.pad(spectrogram, ((0, 0), (0, pad)), "constant")
