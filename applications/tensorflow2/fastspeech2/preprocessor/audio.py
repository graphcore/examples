# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
# Copyright 2020 Minh Nguyen (@dathudeptrai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Graphcore Ltd.
"""
This script has been adapated from the original TensorSpeech/TensorFlowTTS repo found here:
https://github.com/TensorSpeech/TensorFlowTTS/blob/v1.8/tensorflow_tts/bin/preprocess.py

Main changes:
  Remove unused processors.
  Add compute_length function.
  Changed search location for unnormalised data.
"""

import argparse
import glob
import logging
import os
import yaml
import json

import librosa
import numpy as np
import pyworld as pw

from functools import partial
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from text import LJSpeechProcessor, LJSPEECH_SYMBOLS
from utils import remove_outlier
from cleaner import english_cleaners


def parse_and_config():
    """Parse arguments and set configuration parameters."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and text features "
        "(See detail in tensorflow_tts/bin/preprocess_dataset.py)."
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        required=True,
        help="Directory containing the dataset files.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        type=str,
        required=True,
        help="Output directory where features will be saved.",
    )

    parser.add_argument(
        "--config", type=str, required=True, help="YAML format configuration file."
    )
    parser.add_argument(
        "--n_cpus",
        type=int,
        default=4,
        required=False,
        help="Number of CPUs to use in parallel.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.05,
        required=False,
        help="Proportion of files to use as test dataset.",
    )
    parser.add_argument(
        "--seed", type=int, default=1989, help="Random seed to split datasets"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Logging level. 0: DEBUG, 1: INFO and WARNING, 2: INFO, WARNING, and ERROR",
    )
    args = parser.parse_args()

    # set logger
    FORMAT = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    log_level = {0: logging.DEBUG, 1: logging.WARNING, 2: logging.ERROR}
    logging.basicConfig(level=log_level[args.verbose], format=FORMAT)

    # load config
    config = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    config.update(vars(args))
    # config checks
    return config


def ph_based_trim(
        config,
        utt_id: str,
        text_ids: np.array,
        raw_text: str,
        audio: np.array,
        hop_size: int):
    """
    Args:
        config: Parsed yaml config
        utt_id: file name
        text_ids: array with text ids
        raw_text: raw text of file
        audio: parsed wav file
        hop_size: Hop size
    Returns: (bool, np.array, np.array) => if trimmed return True, new text_ids, new audio_array
    """

    os.makedirs(os.path.join(config["rootdir"],
                "trimmed-durations"), exist_ok=True)
    duration_path = config.get(
        "duration_path", os.path.join(config["rootdir"], "durations")
    )
    duration_fixed_path = config.get(
        "duration_fixed_path", os.path.join(
            config["rootdir"], "trimmed-durations")
    )
    sil_ph = ["SIL", "END"]
    text = raw_text.split(" ")

    trim_start, trim_end = False, False

    if text[0] in sil_ph:
        trim_start = True

    if text[-1] in sil_ph:
        trim_end = True

    if not trim_start and not trim_end:
        return False, text_ids, audio

    idx_start, idx_end = (
        0 if not trim_start else 1,
        text_ids.__len__() if not trim_end else -1,
    )
    text_ids = text_ids[idx_start:idx_end]
    durations = np.load(os.path.join(duration_path, f"{utt_id}-durations.npy"))
    if trim_start:
        s_trim = int(durations[0] * hop_size)
        audio = audio[s_trim:]
    if trim_end:
        e_trim = int(durations[-1] * hop_size)
        audio = audio[:-e_trim]

    durations = durations[idx_start:idx_end]
    np.save(os.path.join(duration_fixed_path,
            f"{utt_id}-durations.npy"), durations)
    return True, text_ids, audio


def gen_audio_features(item, config):
    """Generate audio features and transformations
    Args:
        item (Dict): dictionary containing the attributes to encode.
        config (Dict): configuration dictionary.
    Returns:
        (bool): keep this sample or not.
        mel (ndarray): mel matrix in np.float32.
        energy (ndarray): energy audio profile.
        f0 (ndarray): fundamental frequency.
        item (Dict): dictionary containing the updated attributes.
    """
    # get info from sample.
    audio = item["audio"]
    utt_id = item["utt_id"]
    rate = item["rate"]

    # check audio properties
    assert len(audio.shape) == 1, f"{utt_id} seems to be multi-channel signal."
    assert np.abs(audio).max(
    ) <= 1.0, f"{utt_id} is different from 16 bit PCM."

    # check sample rate
    if rate != config["sampling_rate"]:
        audio = librosa.resample(audio, rate, config["sampling_rate"])
        logging.info(
            f"{utt_id} sampling rate is {rate}, not {config['sampling_rate']}, we resample it.")

    # trim silence
    if config["trim_silence"]:
        if "trim_mfa" in config and config["trim_mfa"]:
            _, item["text_ids"], audio = ph_based_trim(
                config,
                utt_id,
                item["text_ids"],
                item["raw_text"],
                audio,
                config["hop_size"],
            )
            if (
                audio.__len__() < 1
            ):  # very short files can get trimmed fully if mfa didnt extract any tokens LibriTTS maybe take only longer files?
                logging.warning(
                    f"File have only silence or MFA didnt extract any token {utt_id}"
                )
                return False, None, None, None, item
        else:
            audio, _ = librosa.effects.trim(
                audio,
                top_db=config["trim_threshold_in_db"],
                frame_length=config["trim_frame_size"],
                hop_length=config["trim_hop_size"],
            )

    # resample audio if necessary
    if "sampling_rate_for_feats" in config:
        audio = librosa.resample(
            audio, rate, config["sampling_rate_for_feats"])
        sampling_rate = config["sampling_rate_for_feats"]
        assert (
            config["hop_size"] * config["sampling_rate_for_feats"] % rate == 0
        ), "'hop_size' must be 'int' value. Please check if 'sampling_rate_for_feats' is correct."
        hop_size = config["hop_size"] * \
            config["sampling_rate_for_feats"] // rate
    else:
        sampling_rate = config["sampling_rate"]
        hop_size = config["hop_size"]

    # get spectrogram
    D = librosa.stft(
        audio,
        n_fft=config["fft_size"],
        hop_length=hop_size,
        win_length=config["win_length"],
        window=config["window"],
        pad_mode="reflect",
    )
    S, _ = librosa.magphase(D)  # (#bins, #frames)

    # get mel basis
    fmin = 0 if config["fmin"] is None else config["fmin"]
    fmax = sampling_rate // 2 if config["fmax"] is None else config["fmax"]
    mel_basis = librosa.filters.mel(
        sr=sampling_rate,
        n_fft=config["fft_size"],
        n_mels=config["num_mels"],
        fmin=fmin,
        fmax=fmax,
    )
    mel = np.log10(np.maximum(np.dot(mel_basis, S), 1e-10)
                   ).T  # (#frames, #bins)

    # check audio and feature length
    audio = np.pad(audio, (0, config["fft_size"]), mode="edge")
    audio = audio[: len(mel) * hop_size]
    assert len(mel) * hop_size == len(audio)

    # extract raw pitch
    _f0, t = pw.dio(
        audio.astype(np.double),
        fs=sampling_rate,
        f0_ceil=fmax,
        frame_period=1000 * hop_size / sampling_rate,
    )
    f0 = pw.stonemask(audio.astype(np.double), _f0, t, sampling_rate)
    if len(f0) >= len(mel):
        f0 = f0[: len(mel)]
    else:
        f0 = np.pad(f0, (0, len(mel) - len(f0)))

    # extract energy
    energy = np.sqrt(np.sum(S ** 2, axis=0))
    assert len(mel) == len(f0) == len(energy)

    # remove outlier f0/energy
    f0 = remove_outlier(f0)
    energy = remove_outlier(energy)

    # apply global gain
    if config["global_gain_scale"] > 0.0:
        audio *= config["global_gain_scale"]
    if np.abs(audio).max() >= 1.0:
        logging.warn(
            f"{utt_id} causes clipping. It is better to reconsider global gain scale value."
        )
    item["audio"] = audio
    item["mel"] = mel
    item["f0"] = f0
    item["energy"] = energy
    return True, mel, energy, f0, item


def save_statistics_to_file(scaler_list, config):
    """Save computed statistics to disk.
    Args:
        scaler_list (List): List of scalers containing statistics to save.
        config (Dict): configuration dictionary.
    """
    for scaler, name in scaler_list:
        stats = np.stack((scaler.mean_, scaler.scale_))
        np.save(
            os.path.join(config["outdir"], f"stats{name}.npy"),
            stats.astype(np.float32),
            allow_pickle=False,
        )


def save_features_to_file(features, subdir, config):
    """Save transformed dataset features in disk.
    Args:
        features (Dict): dictionary containing the attributes to save.
        subdir (str): data split folder where features will be saved.
        config (Dict): configuration dictionary.
    """
    utt_id = features["utt_id"]

    save_list = [
        (features["audio"], "wavs", "wave", np.float32),
        (features["mel"], "raw-feats", "raw-feats", np.float32),
        (features["text_ids"], "ids", "ids", np.int32),
        (features["f0"], "raw-f0", "raw-f0", np.float32),
        (features["energy"], "raw-energies", "raw-energy", np.float32),
    ]
    for item, name_dir, name_file, fmt in save_list:
        np.save(
            os.path.join(
                config["outdir"], subdir, name_dir, f"{utt_id}-{name_file}.npy"
            ),
            item.astype(fmt),
            allow_pickle=False,
        )


def save_vocab_file(config, vocab):
    with open(os.path.join(config["outdir"], "vocab.txt"), "w") as f:
        for v in vocab:
            f.write(v+"\n")


def preprocess(config):
    """Run preprocessing process and compute statistics for normalizing."""
    processor = LJSpeechProcessor(
        config["rootdir"],
        symbols=LJSPEECH_SYMBOLS,
        speakers_map={"ljspeech": 0}
    )

    # check output directories
    def build_dir(x): return [
        os.makedirs(os.path.join(config["outdir"], x, y), exist_ok=True)
        for y in ["raw-feats", "wavs", "ids", "raw-f0", "raw-energies"]
    ]
    build_dir("train")
    build_dir("valid")

    # save pretrained-processor to feature dir
    processor._save_mapper(
        os.path.join(config["outdir"], "ljspeech_mapper.json"),
    )
    save_vocab_file(config, LJSPEECH_SYMBOLS)

    # build train test split
    train_split, valid_split = train_test_split(
        processor.items,
        test_size=config["test_size"],
        random_state=config["seed"],
        shuffle=True,
    )
    logging.info(f"Training items: {len(train_split)}")
    logging.info(f"Validation items: {len(valid_split)}")

    def get_utt_id(x): return os.path.split(x[1])[-1].split(".")[0]
    train_utt_ids = [get_utt_id(x) for x in train_split]
    valid_utt_ids = [get_utt_id(x) for x in valid_split]

    # save train and valid utt_ids to track later
    np.save(os.path.join(config["outdir"], "train_utt_ids.npy"), train_utt_ids)
    np.save(os.path.join(config["outdir"], "valid_utt_ids.npy"), valid_utt_ids)

    # define map iterator
    def iterator_data(items_list):
        for item in items_list:
            yield processor.get_one_sample(item)

    train_iterator_data = iterator_data(train_split)
    valid_iterator_data = iterator_data(valid_split)

    p = Pool(config["n_cpus"])

    # preprocess train files and get statistics for normalizing
    partial_fn = partial(gen_audio_features, config=config)
    train_map = p.imap_unordered(
        partial_fn,
        tqdm(train_iterator_data, total=len(
            train_split), desc="[Preprocessing train]"),
        chunksize=10,
    )
    # init scaler for multiple features
    scaler_mel = StandardScaler(copy=False)
    scaler_energy = StandardScaler(copy=False)
    scaler_f0 = StandardScaler(copy=False)

    id_to_remove = []
    for result, mel, energy, f0, features in train_map:
        if not result:
            id_to_remove.append(features["utt_id"])
            continue
        save_features_to_file(features, "train", config)
        # partial fitting of scalers
        if len(energy[energy != 0]) == 0 or len(f0[f0 != 0]) == 0:
            id_to_remove.append(features["utt_id"])
            continue
        # partial fitting of scalers
        if len(energy[energy != 0]) == 0 or len(f0[f0 != 0]) == 0:
            id_to_remove.append(features["utt_id"])
            continue
        scaler_mel.partial_fit(mel)
        scaler_energy.partial_fit(energy[energy != 0].reshape(-1, 1))
        scaler_f0.partial_fit(f0[f0 != 0].reshape(-1, 1))

    if len(id_to_remove) > 0:
        np.save(
            os.path.join(config["outdir"], "train_utt_ids.npy"),
            [i for i in train_utt_ids if i not in id_to_remove],
        )
        logging.info(
            f"removed {len(id_to_remove)} cause of too many outliers or bad mfa extraction"
        )

    # save statistics to file
    logging.info("Saving computed statistics.")
    scaler_list = [(scaler_mel, ""), (scaler_energy,
                                      "_energy"), (scaler_f0, "_f0")]
    save_statistics_to_file(scaler_list, config)

    # preprocess valid files
    partial_fn = partial(gen_audio_features, config=config)
    valid_map = p.imap_unordered(
        partial_fn,
        tqdm(valid_iterator_data, total=len(
            valid_split), desc="[Preprocessing valid]"),
        chunksize=10,
    )
    for *_, features in valid_map:
        save_features_to_file(features, "valid", config)


def gen_normal_mel(mel_path, scaler, config):
    """Normalize the mel spectrogram and save it to the corresponding path.
    Args:
        mel_path (string): path of the mel spectrogram to normalize.
        scaler (sklearn.base.BaseEstimator): scaling function to use for normalize.
        config (Dict): configuration dictionary.
    """
    mel = np.load(mel_path)
    mel_norm = scaler.transform(mel)
    path, file_name = os.path.split(mel_path)
    *_, subdir, suffix = path.split(os.sep)

    utt_id = file_name.split(f"-{suffix}.npy")[0]
    np.save(
        os.path.join(
            config["outdir"], subdir, "norm-feats", f"{utt_id}-norm-feats.npy"
        ),
        mel_norm.astype(np.float32),
        allow_pickle=False,
    )


def normalize(config):
    """Normalize mel spectrogram with pre-computed statistics."""
    # init scaler with saved values
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = np.load(
        os.path.join(config["outdir"], "stats.npy")
    )
    scaler.n_features_in_ = config["num_mels"]

    # find all "raw-feats" files in both train and valid folders
    glob_path = os.path.join(config["outdir"], "**", "raw-feats", "*.npy")
    mel_raw_feats = glob.glob(glob_path, recursive=True)
    logging.info(f"Files to normalize: {len(mel_raw_feats)}")

    # check for output directories
    os.makedirs(os.path.join(config["outdir"],
                "train", "norm-feats"), exist_ok=True)
    os.makedirs(os.path.join(config["outdir"],
                "valid", "norm-feats"), exist_ok=True)

    p = Pool(config["n_cpus"])
    partial_fn = partial(gen_normal_mel, scaler=scaler, config=config)
    list(p.map(partial_fn, tqdm(mel_raw_feats, desc="[Normalizing]")))


def compute_statistics(config):
    """Compute mean / std statistics of some features for later normalization."""

    # find features files for the train split
    def glob_fn(x): return glob.glob(os.path.join(
        config["rootdir"], "train", x, "*.npy"))
    glob_ids = glob_fn("ids")
    glob_mel = glob_fn("raw-feats")
    glob_f0 = glob_fn("raw-f0")
    glob_energy = glob_fn("raw-energies")
    assert (
        len(glob_mel) == len(glob_f0) == len(glob_energy)
    ), "Features, f0 and energies have different files in training split."

    logging.info(f"Computing statistics for {len(glob_mel)} files.")
    # init scaler for multiple features
    scaler_mel = StandardScaler(copy=False)
    scaler_energy = StandardScaler(copy=False)
    scaler_f0 = StandardScaler(copy=False)

    for input_id, mel, f0, energy in tqdm(
        zip(glob_ids, glob_mel, glob_f0, glob_energy), total=len(glob_mel)
    ):
        ids = np.load(input_id)
        energy = np.load(energy)
        f0 = np.load(f0)
        melspec = np.load(mel)
        max_seq_length = max(max_seq_length, len(ids))
        max_mel_length = max(max_mel_length, melspec.shape[0])
        # partial fitting of scalers
        scaler_mel.partial_fit(np.load(mel))
        scaler_energy.partial_fit(energy[energy != 0].reshape(-1, 1))
        scaler_f0.partial_fit(f0[f0 != 0].reshape(-1, 1))

    # save statistics to file
    logging.info("Saving computed statistics.")
    scaler_list = [(scaler_mel, ""), (scaler_energy,
                                      "_energy"), (scaler_f0, "_f0")]
    save_statistics_to_file(scaler_list, config)


def compute_length(config):
    """Compute max input sequence length and mel-spectrum length in datasets."""
    def glob_fn(n, x): return glob.glob(
        os.path.join(config["outdir"], n, x, "*.npy"))

    seq_length = {}
    mel_length = {}
    for fn in ["train", "valid"]:
        seq_length[fn] = 0
        mel_length[fn] = 0
        logging.info(f"Computing [{fn}] length of mel-spectrograms and ids.")
        for input_id, mel in tqdm(zip(glob_fn(fn, "ids"), glob_fn(fn, "raw-feats"))):
            ids = np.load(input_id)
            melspec = np.load(mel)
            seq_length[fn] = max(seq_length[fn], len(ids))
            mel_length[fn] = max(mel_length[fn], len(melspec))

    max_seq_length = max(seq_length.values())
    max_mel_length = max(mel_length.values())
    length_mapper = {
        "max_seq_length": max_seq_length,
        "max_mel_length": max_mel_length,
        "vocab_size": len(LJSPEECH_SYMBOLS)
    }
    with open(os.path.join(config["outdir"], "length.json"), "w") as f:
        json.dump(length_mapper, f)


if __name__ == "__main__":
    config = parse_and_config()
    preprocess(config)
    normalize(config)
    compute_length(config)
