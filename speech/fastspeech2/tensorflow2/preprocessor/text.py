# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
# Copyright 2020 TensorFlowTTS Team.
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
[
  https://github.com/TensorSpeech/TensorFlowTTS/blob/v1.8/tensorflow_tts/processor/base_processor.py,
  https://github.com/TensorSpeech/TensorFlowTTS/blob/v1.8/tensorflow_tts/processor/ljspeech.py
]

Main changes:
  Combine two files.
"""

import os
import re
import abc
import json
import numpy as np
import soundfile as sf
from typing import Dict, List, Union
from dataclasses import dataclass, field
from cleaner import english_cleaners


class DataProcessorError(Exception):
    pass


valid_symbols = [
    "AA",
    "AA0",
    "AA1",
    "AA2",
    "AE",
    "AE0",
    "AE1",
    "AE2",
    "AH",
    "AH0",
    "AH1",
    "AH2",
    "AO",
    "AO0",
    "AO1",
    "AO2",
    "AW",
    "AW0",
    "AW1",
    "AW2",
    "AY",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "EH0",
    "EH1",
    "EH2",
    "ER",
    "ER0",
    "ER1",
    "ER2",
    "EY",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH",
    "IH0",
    "IH1",
    "IH2",
    "IY",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OW0",
    "OW1",
    "OW2",
    "OY",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UH0",
    "UH1",
    "UH2",
    "UW",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]

_pad = "pad"
_eos = "eos"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in valid_symbols]

# Export all symbols:
LJSPEECH_SYMBOLS = (
    [_pad] + list(_special) + list(_punctuation) +
    list(_letters) + _arpabet + [_eos]
)

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


@dataclass
class BaseProcessor(abc.ABC):
    """Base Processor for all processor."""
    data_dir: str
    symbols: List[str]
    speakers_map: Dict[str, int]

    train_f_name: str = "train.txt"
    delimiter: str = "|"
    positions = {
        "file": 0,
        "text": 1,
        "speaker_name": 2,
    }  # positions of file,text,speaker_name after split line
    f_extension: str = ".wav"
    saved_mapper_path: str = None
    loaded_mapper_path: str = None
    # extras
    # text, wav_path, speaker_name
    items: List[List[str]] = field(default_factory=list)
    symbol_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_symbol: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self):

        if self.loaded_mapper_path is not None:
            self._load_mapper(loaded_path=self.loaded_mapper_path)
            if self.setup_eos_token():
                self.add_symbol(
                    self.setup_eos_token()
                )  # if this eos token not yet present in symbols list.
                self.eos_id = self.symbol_to_id[self.setup_eos_token()]
            return

        if self.symbols.__len__() < 1:
            raise DataProcessorError(
                "Symbols list is empty but mapper isn't loaded")

        self.create_items()
        self.create_speaker_map()
        self.reverse_speaker = {v: k for k, v in self.speakers_map.items()}
        self.create_symbols()
        if self.saved_mapper_path is not None:
            self._save_mapper(saved_path=self.saved_mapper_path)

        # processor name. useful to use it for AutoProcessor
        self._processor_name = type(self).__name__

        if self.setup_eos_token():
            self.add_symbol(
                self.setup_eos_token()
            )  # if this eos token not yet present in symbols list.
            self.eos_id = self.symbol_to_id[self.setup_eos_token()]

    def __getattr__(self, name: str) -> Union[str, int]:
        if "_id" in name:  # map symbol to id
            return self.symbol_to_id[name.replace("_id", "")]
        return self.symbol_to_id[name]  # map symbol to value

    def create_speaker_map(self):
        """
        Create speaker map for dataset.
        """
        sp_id = 0
        for i in self.items:
            speaker_name = i[-1]
            if speaker_name not in self.speakers_map:
                self.speakers_map[speaker_name] = sp_id
                sp_id += 1

    def get_speaker_id(self, name: str) -> int:
        return self.speakers_map[name]

    def get_speaker_name(self, speaker_id: int) -> str:
        return self.speakers_map[speaker_id]

    def create_symbols(self):
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def create_items(self):
        """
        Method used to create items from training file
        items struct example => text, wav_file_path, speaker_name.
        Note that the speaker_name should be a last.
        """
        with open(
            os.path.join(self.data_dir, self.train_f_name), mode="r", encoding="utf-8"
        ) as f:
            for line in f:
                parts = line.strip().split(self.delimiter)
                wav_path = os.path.join(
                    self.data_dir, parts[self.positions["file"]])
                wav_path = (
                    wav_path + self.f_extension
                    if wav_path[-len(self.f_extension):] != self.f_extension
                    else wav_path
                )
                text = parts[self.positions["text"]]
                speaker_name = parts[self.positions["speaker_name"]]
                self.items.append([text, wav_path, speaker_name])

    def add_symbol(self, symbol: Union[str, list]):
        if isinstance(symbol, str):
            if symbol in self.symbol_to_id:
                return
            self.symbols.append(symbol)
            symbol_id = len(self.symbol_to_id)
            self.symbol_to_id[symbol] = symbol_id
            self.id_to_symbol[symbol_id] = symbol

        elif isinstance(symbol, list):
            for i in symbol:
                self.add_symbol(i)
        else:
            raise ValueError(
                "A new_symbols must be a string or list of string.")

    @abc.abstractmethod
    def get_one_sample(self, item):
        """Get one sample from dataset items.
        Args:
            item: one item in Dataset items.
                Dataset items may include (raw_text, speaker_id, wav_path, ...)
        Returns:
            sample (dict): sample dictionary return all feature used for preprocessing later.
        """
        sample = {
            "raw_text": None,
            "text_ids": None,
            "audio": None,
            "utt_id": None,
            "speaker_name": None,
            "rate": None,
        }
        return sample

    @abc.abstractmethod
    def text_to_sequence(self, text: str):
        return []

    @abc.abstractmethod
    def setup_eos_token(self):
        """Return eos symbol of type string."""
        return "eos"

    def convert_symbols_to_ids(self, symbols: Union[str, list]):
        sequence = []
        if isinstance(symbols, str):
            sequence.append(self._symbol_to_id[symbols])
            return sequence
        elif isinstance(symbols, list):
            for s in symbols:
                if isinstance(s, str):
                    sequence.append(self._symbol_to_id[s])
                else:
                    raise ValueError(
                        "All elements of symbols must be a string.")
        else:
            raise ValueError("A symbols must be a string or list of string.")

        return sequence

    def _load_mapper(self, loaded_path: str = None):
        """
        Save all needed mappers to file
        """
        loaded_path = (
            os.path.join(self.data_dir, "mapper.json")
            if loaded_path is None
            else loaded_path
        )
        with open(loaded_path, "r") as f:
            data = json.load(f)
        self.speakers_map = data["speakers_map"]
        self.symbol_to_id = data["symbol_to_id"]
        self.id_to_symbol = {
            int(k): v for k, v in data["id_to_symbol"].items()}
        self._processor_name = data["processor_name"]

        # other keys
        all_data_keys = data.keys()
        for key in all_data_keys:
            if key not in ["speakers_map", "symbol_to_id", "id_to_symbol"]:
                setattr(self, key, data[key])

    def _save_mapper(self, saved_path: str = None, extra_attrs_to_save: dict = None):
        """
        Save all needed mappers to file
        """
        saved_path = (
            os.path.join(self.data_dir, "mapper.json")
            if saved_path is None
            else saved_path
        )
        with open(saved_path, "w") as f:
            full_mapper = {
                "symbol_to_id": self.symbol_to_id,
                "id_to_symbol": self.id_to_symbol,
                "speakers_map": self.speakers_map,
                "processor_name": self._processor_name,
            }
            if extra_attrs_to_save:
                full_mapper = {**full_mapper, **extra_attrs_to_save}
            json.dump(full_mapper, f)

    @abc.abstractmethod
    def save_pretrained(self, saved_path):
        """Save mappers to file"""
        pass


@dataclass
class LJSpeechProcessor(BaseProcessor):
    """LJSpeech processor."""

    positions = {
        "wave_file": 0,
        "text": 1,
        "text_norm": 2,
    }
    train_f_name: str = "metadata.csv"

    def create_items(self):
        if self.data_dir:
            with open(
                os.path.join(self.data_dir, self.train_f_name), encoding="utf-8"
            ) as f:
                self.items = [self.split_line(
                    self.data_dir, line, "|") for line in f]

    def split_line(self, data_dir, line, split):
        parts = line.strip().split(split)
        wave_file = parts[self.positions["wave_file"]]
        text_norm = parts[self.positions["text_norm"]]
        wav_path = os.path.join(data_dir, "wavs", f"{wave_file}.wav")
        speaker_name = "ljspeech"
        return text_norm, wav_path, speaker_name

    def setup_eos_token(self):
        return _eos

    def save_pretrained(self, saved_path):
        os.makedirs(saved_path, exist_ok=True)
        self._save_mapper(os.path.join(saved_path, "processor.json"), {})

    def get_one_sample(self, item):
        text, wav_path, speaker_name = item

        # normalize audio signal to be [-1, 1], soundfile already norm.
        audio, rate = sf.read(wav_path)
        audio = audio.astype(np.float32)

        # convert text to ids
        text_ids = np.asarray(self.text_to_sequence(text), np.int32)

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": os.path.split(wav_path)[-1].split(".")[0],
            "speaker_name": speaker_name,
            "rate": rate,
        }

        return sample

    def text_to_sequence(self, text):
        sequence = []
        # Check for curly braces and treat their contents as ARPAbet:
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += self._symbols_to_sequence(
                    english_cleaners(text)
                )
                break
            sequence += self._symbols_to_sequence(
                english_cleaners(m.group(1))
            )
            sequence += self._arpabet_to_sequence(m.group(2))
            text = m.group(3)

        # add eos tokens
        sequence += [self.eos_id]
        return sequence

    def _symbols_to_sequence(self, symbols):
        return [self.symbol_to_id[s] for s in symbols if self._should_keep_symbol(s)]

    def _arpabet_to_sequence(self, text):
        return self._symbols_to_sequence(["@" + s for s in text.split()])

    def _should_keep_symbol(self, s):
        return s in self.symbol_to_id and s != "_" and s != "~"
