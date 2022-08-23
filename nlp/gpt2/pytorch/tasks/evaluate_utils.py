# coding=utf-8
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Graphcore Ltd.

import json
import math

import torch
import numpy as np


def process_batch(batch):
    """Process batch and produce inputs for the model."""

    loss_mask = batch['pad_mask'].long().contiguous().byte()
    tokens_ = batch['text'].long().contiguous()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    return tokens, labels, loss_mask


class _LMDataset(torch.utils.data.Dataset):

    def __init__(self, tokens, seq_len, pad_idx, num_original_tokens,
                 num_tokenized_tokens, overlapping_eval=None):
        self.tokens = tokens
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.overlapping_eval = overlapping_eval
        if self.overlapping_eval is None:
            self.overlapping_eval = self.seq_len
        self.overlapping_eval = max(1, self.overlapping_eval)
        self.num_original_tokens = num_original_tokens
        self.num_tokenized_tokens = num_tokenized_tokens
        self.total_targets = len(self.tokens) - 1
        # remove first sequence tokens
        targets = max(self.total_targets - self.overlapping_eval, 0)
        self.total_sequences = max(
            math.ceil(targets / self.overlapping_eval) + 1, 1)

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.overlapping_eval
        end_idx = start_idx + self.seq_len
        tokens = self.tokens[start_idx:end_idx + 1]
        num_tokens = len(tokens)
        pad_mask = [1] * num_tokens
        if num_tokens < self.seq_len + 1:
            num_pad = (self.seq_len + 1 - num_tokens)
            pad_mask += [0] * (num_pad)
            tokens += [self.pad_idx] * num_pad
        pad_mask = np.array(pad_mask[1:])
        if self.overlapping_eval != self.seq_len and idx != 0:
            pad_mask[:-self.overlapping_eval] *= 0

        return {'text': np.array(tokens), 'pad_mask': pad_mask}


class _LambadaDataset(torch.utils.data.Dataset):

    def __init__(self, path, pad_idx, tokenizer, seq_len, strict=False):
        print('> building lambada dataset from {} ...'.format(path))
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.tokenizer = tokenizer
        self.strict = strict

        self.tokens = []
        self.labels = []
        with open(path, 'r') as f:
            for line in f.readlines():
                text = json.loads(line)['text']
                tokens, labels = self.get_tokens(text)
                if tokens:
                    self.tokens.append(tokens)
                    self.labels.append(labels)

    def get_tokens(self, text):
        if not self.strict:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len]
            return tokens[:-1], [tokens[-1]]
        last_token = text.split()[-1]
        start_idx = text.rfind(last_token)
        beginning_tokens = self.tokenizer.encode(text[:start_idx].strip())
        last_token = self.tokenizer.encode(' ' + last_token)
        if len(beginning_tokens) + len(last_token) > self.seq_len:
            return None, None
        return beginning_tokens, last_token

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        num_tokens = len(tokens)
        pad_mask = [0] * num_tokens
        labels = self.labels[idx]
        pad_mask += [1] * len(labels)
        tokens = tokens + labels
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = (self.seq_len + 1 - num_tokens)
            pad_mask += [0] * (num_pad)
            tokens += [self.pad_idx] * num_pad
        pad_mask = np.array(pad_mask[1:])

        return {'text': np.array(tokens), 'pad_mask': pad_mask}
