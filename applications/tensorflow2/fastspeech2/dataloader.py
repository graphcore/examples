# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
# Copyright 2020 Minh Nguyen (@dathudeptrai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Graphcore Ltd.
"""
The function `average_by_duration`and `_norm_mean_std` were copied from
https://github.com/TensorSpeech/TensorFlowTTS/blob/v1.8/examples/fastspeech2/fastspeech2_dataset.py
"""
import os
import json
import logging
import numpy as np
import tensorflow as tf


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)06d: %(levelname)-1.1s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def average_by_duration(x, durs):
    durs = durs.astype(np.int32)
    mel_len = durs.sum()
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))  # pad 0 to the start

    # calculate charactor f0/energy
    x_char = np.zeros((durs.shape[0],), dtype=np.float32)
    for idx, start, end in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = x[start:end][np.where(x[start:end] != 0.0)[0]]
        x_char[idx] = np.mean(values) if len(values) > 0 else 0.0
    return x_char.astype(np.float32)


class LJSpeechDataset(object):
    """Dataloader for phoneme-level datasets."""

    def __init__(self, opts, is_train=True):
        self.opts = opts
        self.is_train = is_train
        self.dtype = tf.float16 if opts["precision"] == "16" else tf.float32
        self.np_dtype = np.float16 if opts["precision"] == "16" else np.float32
        self.max_seq_len = opts["max_seq_length"]
        self.max_mel_length = opts["max_wave_length"]
        self.filenames = []
        if not self.opts["generated_data"] and self.opts["data_path"]:
            files_path = os.path.join(self.opts["data_path"], "train.txt") if self.is_train else os.path.join(
                self.opts["data_path"], "val.txt")
            with open(os.path.abspath(files_path)) as fp:
                for line in fp.readlines():
                    self.filenames.append(line.strip())
            if is_train:
                np.random.shuffle(self.filenames)

    def __len__(self):
        if self.opts["generated_data"]:
            return 10000
        return len(self.filenames)

    def _load_data(self, filename):
        phoneme = np.load(os.path.join(
            self.opts["data_path"], "phone", f"phn-{filename}.npy")).astype(np.int32)
        duration = np.load(os.path.join(
            self.opts["data_path"], "duration", f"duration-{filename}.npy")).astype(self.np_dtype)
        pitch = np.load(os.path.join(
            self.opts["data_path"], "pitch", f"pitch-{filename}.npy")).astype(self.np_dtype)
        energy = np.load(os.path.join(
            self.opts["data_path"], "energy", f"energy-{filename}.npy")).astype(self.np_dtype)
        mel = np.load(os.path.join(
            self.opts["data_path"], "mel", f"mel-{filename}.npy")).astype(self.np_dtype)
        return phoneme, duration, pitch, energy, mel

    def _fake_duration(self, phn_len, mel_len):
        dur = [mel_len//phn_len]*phn_len
        balance = sum(dur) - mel_len
        dur[-1] += balance
        return np.array(dur).astype(np.int32)

    def _generated_generator(self):
        while True:
            phoneme = np.random.randint(0, self.max_seq_len,
                                        size=(self.max_seq_len,)).astype(np.int32)
            duration = self._fake_duration(
                self.max_seq_len, self.max_mel_length)
            mel = np.random.rand(self.max_mel_length, self.opts["num_mels"]).astype(
                self.np_dtype)
            pitch = np.random.rand(self.max_seq_len,).astype(self.np_dtype)
            energy = np.random.rand(self.max_seq_len,).astype(self.np_dtype)

            yield phoneme, duration, pitch, energy, mel

    def _inference_generator(self):
        while True:
            for fn in self.filenames:
                phoneme = np.load(os.path.join(
                    self.opts["data_path"], "phone", f"phn-{fn}.npy")).astype(np.int32)
                yield phoneme

    def generator(self):
        while True:
            for fn in self.filenames:
                phoneme, duration, pitch, energy, mel = self._load_data(fn)
                yield phoneme, duration, pitch, energy, mel

    def __call__(self):
        """Create tf.dataset function."""
        tf.random.set_seed(int(self.opts['seed']))
        np.random.seed(int(self.opts['seed']))

        output_types = (tf.int32, self.dtype, self.dtype,
                        self.dtype, self.dtype)
        padded_shapes = ([self.max_seq_len], [self.max_seq_len], [self.max_seq_len], [
                         self.max_seq_len], [self.max_mel_length, self.opts["num_mels"]])

        if self.opts["generated_data"]:
            data_gen = self._generated_generator
        else:
            if not self.filenames:
                self._shuffle_files()
            data_gen = self.generator

        datasets = tf.data.Dataset.from_generator(
            data_gen, output_types=output_types)
        if self.is_train:
            datasets = datasets.shuffle(
                buffer_size=1000, seed=int(self.opts["seed"]))
        datasets = datasets.padded_batch(
            self.opts["batch_size"], padded_shapes=padded_shapes, drop_remainder=True)
        datasets = datasets.map(lambda phoneme, duration, pitch, energy, melspectrum: (
            (phoneme, duration, pitch, energy), (melspectrum, melspectrum, duration, pitch, energy)))
        datasets = datasets.repeat().prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_inference_data(self):
        """Create tf.dataset function."""
        tf.random.set_seed(int(self.opts['seed']))
        np.random.seed(int(self.opts['seed']))

        output_types = (tf.int32)
        padded_shapes = ([self.max_seq_len])

        datasets = tf.data.Dataset.from_generator(
            self._inference_generator, output_types=output_types)
        datasets = datasets.padded_batch(
            self.opts["batch_size"], padded_shapes=padded_shapes, drop_remainder=True)
        datasets = datasets.repeat(1).prefetch(tf.data.experimental.AUTOTUNE)
        return datasets


class LJSpeechCharLevelDataset(object):
    """Dataloader for character-level datasets."""

    def __init__(self, opts, is_train=True):
        self.opts = opts
        self.is_train = is_train
        self.np_dtype = np.float16 if opts["precision"] == "16" else np.float32
        self.max_seq_length = opts["max_seq_length"]
        self.max_mel_length = opts["max_wave_length"]
        self.dtype = tf.float16 if opts["precision"] == "16" else tf.float32
        self.np_dtype = np.float16 if opts["precision"] == "16" else np.float32
        if not self.opts["generated_data"] and self.opts["data_path"]:
            self.utts_path = os.path.join(opts["data_path"], "train_utt_ids.npy") if is_train else os.path.join(
                opts["data_path"], "valid_utt_ids.npy")
            self.base_path = os.path.join(
                opts["data_path"], "train") if is_train else os.path.join(opts["data_path"], "valid")
            self.utts_ids = np.load(self.utts_path)
            # stats
            self.f0_stat = np.load(os.path.join(
                opts["data_path"], "stats_f0.npy"))
            self.energy_stat = np.load(os.path.join(
                opts["data_path"], "stats_energy.npy"))
            self.mel_stat = np.load(os.path.join(
                opts["data_path"], "stats.npy"))
            self._set_path()
            self._get_length()
        self.same_sample = False

    def _set_path(self):
        self.duration_path = os.path.join(self.base_path, "duration")
        self.id_path = os.path.join(self.base_path, "ids")
        self.mel_path = os.path.join(self.base_path, "norm-feats")
        self.f0_path = os.path.join(self.base_path, "raw-f0")
        self.energy_path = os.path.join(self.base_path, "raw-energies")

    def _get_length(self):
        with open(os.path.join(self.opts["data_path"], "length.json"), "r") as f:
            length = json.load(f)
        self.max_seq_length = length["max_seq_length"]
        self.max_mel_length = length["max_mel_length"]

    def __len__(self):
        return len(self.utts_ids)

    def _load_data(self, utt_id):
        input_id = np.load(os.path.join(
            self.id_path, f"{utt_id}-ids.npy")).astype(np.int32)
        f0 = np.load(os.path.join(
            self.f0_path, f"{utt_id}-raw-f0.npy")).astype(self.np_dtype)
        energy = np.load(os.path.join(self.energy_path,
                         f"{utt_id}-raw-energy.npy")).astype(self.np_dtype)
        duration = np.load(os.path.join(self.duration_path,
                           f"{utt_id}-durations.npy")).astype(self.np_dtype)
        mel = np.load(os.path.join(
            self.mel_path, f"{utt_id}-norm-feats.npy")).astype(self.np_dtype)

        assert len(f0) == len(energy) == mel.shape[0], \
            f"[{utt_id}]Shape mismatch!(f0({f0.shape}), energy({energy.shape}) and mel({mel.shape[0]})"
        assert sum(duration) == mel.shape[0], \
            f"[{utt_id}]Sum of duration({sum(duration)}) is not equal to mel.shape[0]({mel.shape[0]})."

        f0 = self._norm_mean_std(f0, self.f0_stat[0], self.f0_stat[1])
        energy = self._norm_mean_std(
            energy, self.energy_stat[0], self.energy_stat[1]
        )

        # calculate charactor f0/energy
        f0 = average_by_duration(f0, duration)
        energy = average_by_duration(energy, duration)
        return input_id, duration, f0, energy, mel

    def _norm_mean_std(self, x, mean, std):
        zero_idxs = np.where(x == 0.0)[0]
        x = (x - mean) / std
        x[zero_idxs] = 0.0
        return x

    def _fake_duration(self, phn_len, mel_len):
        # duration will be padded during loading.
        dur = [mel_len//phn_len]*(phn_len-1)
        balance = sum(dur) - mel_len
        dur[-1] += balance
        return np.array(dur).astype(np.int32)

    def _generated_generator(self):
        while True:
            input_id = np.random.randint(0, self.max_seq_length,
                                         size=(self.max_seq_length,)).astype(np.int32)
            duration = self._fake_duration(
                self.max_seq_length, self.max_mel_length)
            mel = np.random.rand(self.max_mel_length,
                                 self.opts["num_mels"]).astype(self.np_dtype)
            f0 = np.random.rand(self.max_seq_length,).astype(self.np_dtype)
            energy = np.random.rand(self.max_seq_length,).astype(self.np_dtype)
            yield input_id, duration, f0, energy, mel

    def _inference_generator(self):
        for utt_id in self.utts_ids:
            input_id = np.load(os.path.join(
                self.id_path, f"{utt_id}-ids.npy")).astype(np.int32)
            yield input_id

    def generator(self):
        if self.same_sample:
            uid = self.utts_ids[0]
            input_id, duration, f0, energy, mel = self._load_data(uid)
            while True:
                logger.info(
                    f"[Same samples({uid})]uid={uid}, id_gt.shape={input_id.shape}, duration_gt.shape={duration.shape}, f0_gt.shape={f0.shape}, mel_gt.shape={mel.shape}, sum_duration={np.sum(duration)}")
                yield input_id, duration, f0, energy, mel
        else:
            while True:
                for uid in self.utts_ids:
                    try:
                        input_id, duration, f0, energy, mel = self._load_data(
                            uid)
                        yield input_id, duration, f0, energy, mel
                    except AssertionError:
                        pass

    def get_one_samples(self):
        uid = np.random.choice(self.utts_ids)
        input_id, duration, f0, energy, mel = self._load_data(uid)
        return uid, input_id, duration, f0, energy, mel

    def get_inference_data(self):
        """Create tf.dataset function."""
        tf.random.set_seed(int(self.opts['seed']))
        np.random.seed(int(self.opts['seed']))

        output_types = (tf.int32)
        padded_shapes = ([self.max_seq_length])

        datasets = tf.data.Dataset.from_generator(
            self._inference_generator, output_types=output_types)
        datasets = datasets.padded_batch(
            self.opts["batch_size"], padded_shapes=padded_shapes, drop_remainder=True)
        datasets = datasets.repeat(1).prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def __call__(self):
        """Create tf.dataset function."""
        tf.random.set_seed(int(self.opts['seed']))
        np.random.seed(int(self.opts['seed']))

        output_types = (tf.int32, self.dtype, self.dtype,
                        self.dtype, self.dtype)
        padded_shapes = ([self.max_seq_length], [self.max_seq_length], [self.max_seq_length], [
                         self.max_seq_length], [self.max_mel_length, self.opts["num_mels"]])

        if self.opts["generated_data"] or not self.opts["data_path"]:
            data_gen = self._generated_generator
        else:
            data_gen = self.generator

        datasets = tf.data.Dataset.from_generator(
            data_gen, output_types=output_types)
        if self.is_train:
            datasets = datasets.shuffle(
                buffer_size=1000, seed=int(self.opts["seed"]))
        datasets = datasets.padded_batch(
            self.opts["batch_size"], padded_shapes=padded_shapes, drop_remainder=True)

        datasets = datasets.map(lambda input_id, duration, pitch, energy, melspectrum: (
            (input_id, duration, pitch, energy), (melspectrum, melspectrum, duration, pitch, energy)))

        datasets = datasets.repeat().prefetch(tf.data.experimental.AUTOTUNE)
        return datasets


if __name__ == "__main__":
    from options import make_global_options
    opts = make_global_options([])
    train_datasets = LJSpeechCharLevelDataset(opts, is_train=True)
    val_datasets = LJSpeechCharLevelDataset(opts, is_train=False)
    print(
        f"Train datasets: {len(train_datasets)}, Valid datasets: {len(val_datasets)}")
    traindata = train_datasets()
    valdata = val_datasets()
    (input_id, duration, f0, energy, mel), y = next(iter(traindata))
    print("******* Train datasets:")
    print(f"input_id: shape={input_id.shape}, dtype={input_id.dtype}")
    print(f"duration: shape={duration.shape}, dtype={duration.dtype}")
    print(f"f0: shape={f0.shape}, dtype={f0.dtype}")
    print(f"energy: shape={energy.shape}, dtype={energy.dtype}")
    print(f"mel: shape={mel.shape}, dtype={mel.dtype}")
