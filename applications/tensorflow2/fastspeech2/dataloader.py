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
from multiprocessing import Pool, cpu_count


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


class LJSpeechCharLevelDataset(object):
    """Dataloader for character-level datasets."""

    def __init__(self, opts, is_train=True):
        self.opts = opts
        self.is_train = is_train
        self.max_seq_length = opts["max_seq_length"]
        self.max_mel_length = opts["max_wave_length"]
        self.dtype = tf.float16 if opts["precision"] == "16" else tf.float32
        self.np_dtype = np.float16 if opts["precision"] == "16" else np.float32
        if not self.opts["generated_data"] and self.opts["data_path"]:
            utts_path = os.path.join(opts["data_path"], "train_utt_ids.npy") if is_train else os.path.join(
                opts["data_path"], "valid_utt_ids.npy")
            self.base_path = os.path.join(
                opts["data_path"], "train") if is_train else os.path.join(opts["data_path"], "valid")
            self.utts_ids = np.load(utts_path)
            # stats
            self.f0_stat = np.load(os.path.join(
                opts["data_path"], "stats_f0.npy"))
            self.energy_stat = np.load(os.path.join(
                opts["data_path"], "stats_energy.npy"))
            self.mel_stat = np.load(os.path.join(
                opts["data_path"], "stats.npy"))
            self._set_path()
            self._get_length()
            # load data to memory initially to speed up throughput
            with Pool(cpu_count()) as p:
                self.train_data = p.map(self._load_data, self.utts_ids)

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
        if self.opts["generated_data"]:
            return 1000
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
        try:
            # drop the shape mismatched data
            assert len(f0) == len(energy) == mel.shape[0], \
                f"[{utt_id}]Shape mismatch!(f0({f0.shape}), energy({energy.shape}) and mel({mel.shape[0]})"
            assert sum(duration) == mel.shape[0], \
                f"[{utt_id}]Sum of duration({sum(duration)}) is not equal to mel.shape[0]({mel.shape[0]})."
        except AssertionError as e:
            pass
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
        # generated random data once and feed to model repeatedly.
        input_id = np.random.randint(0, self.max_seq_length,
                                     size=(self.max_seq_length,)).astype(np.int32)
        duration = self._fake_duration(
            self.max_seq_length, self.max_mel_length)
        mel = np.random.rand(self.max_mel_length,
                             self.opts["num_mels"]).astype(self.np_dtype)
        f0 = np.random.rand(self.max_seq_length,).astype(self.np_dtype)
        energy = np.random.rand(self.max_seq_length,).astype(self.np_dtype)
        while True:
            yield input_id, duration, f0, energy, mel

    def generator(self):
        while True:
            for input_id, duration, f0, energy, mel in self.train_data:
                yield input_id, duration, f0, energy, mel

    def inference_generator(self):
        while True:
            if self.opts["generated_data"]:
                input_id = np.random.randint(0, self.max_seq_length,
                                             size=(self.max_seq_length,)).astype(np.int32)
                yield input_id
            else:
                for uid in self.utts_ids:
                    try:
                        input_id_file = os.path.join(
                            self.id_path, f"{uid}-ids.npy")
                        input_id = np.load(input_id_file).astype(np.int32)
                        yield input_id
                    except AssertionError:
                        pass

    def get_inference_data(self):
        datasets = tf.data.Dataset.from_generator(
            self.inference_generator, output_types=(tf.int32))
        datasets = datasets.padded_batch(self.opts["batch_size"], padded_shapes=([
                                         self.max_seq_length]), drop_remainder=True)
        datasets = datasets.repeat().prefetch(tf.data.experimental.AUTOTUNE)
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

        datasets = datasets.map(
            lambda input_id, duration, pitch, energy, melspectrum: (
                (input_id, duration, pitch, energy), (melspectrum, melspectrum, duration, pitch, energy)),
            num_parallel_calls=64)

        datasets = datasets.repeat().prefetch(tf.data.experimental.AUTOTUNE)
        return datasets


if __name__ == "__main__":
    import json
    from tensorflow.python.ipu.dataset_benchmark import dataset_benchmark
    from options import make_global_options
    opts = make_global_options([])
    train_datasets = LJSpeechCharLevelDataset(opts, is_train=True)
    json_string = dataset_benchmark(train_datasets(), number_of_epochs=opts["epochs"], elements_per_epochs=int(
        len(train_datasets)/opts["batch_size"])).numpy()
    json_object = json.loads(json_string[0].decode('utf-8'))
    mean_throughput = np.mean([epoch['elements_per_second']
                              for epoch in json_object["epochs"]]) * opts["batch_size"]
    print(f"Mean throughput: {mean_throughput:.2f} samples/s")
