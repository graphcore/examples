# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

import multiprocessing
import random
from collections import deque

import numpy as np
try:
    from torch_xla.utils.tf_record_reader import TfRecordReader
except ImportError:
    raise ImportError("""Torch-xla required for TFRecord dataset.
                      Please install torch 1.7.0 & torch-xla using
                     `pip install torch==1.7.0 torch-xla@https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.7-cp36-cp36m-linux_x86_64.whl`""")

KEYS = ('masked_lm_ids', 'masked_lm_weights', 'segment_ids',
        'input_ids', 'input_mask', 'next_sentence_labels',
        'masked_lm_positions')


class PretrainingTfRecordDataLoader:
    """Iterates through TFrecord files to generate batch worth of data.
    Each iteration returns a list of 7 numpy arrays, consisting of
    return_value[0] Token indices
                      shape (batch_size x sequence_length),
                      each item taking values from [0, vocab_length)
    return_value[1] Position indices
                      shape (batch_size x sequence_length)
                      each item taking values from [0, sequence_length)
    return_value[2] Segment indices
                      shape (batch_size x sequence_length)
                      each item taking values from [0, 2)
    return_value[3] Mask token idx
                      shape (batch_size x 1)
                      in [1, num_mask_tokens]
                      indicates number of actual mask tokens.
    return_value[4] Sequence length idx
                      shape (batch_size x 1)
                      in [1, sequence_length]
                      indicates actual sequence length.
    return_value[5] Mask labels
                      shape (batch_size x num_mask_tokens)
                      each item taking values from [0, vocab_length)
    return_value[6] Next Sentence Prediction labels
                      shape (batch_size x 1)
                      each item taking values from [0, 3)
    """
    def __init__(self,
                 input_files,
                 max_seq_length,
                 max_mask_tokens,
                 batch_size=1,
                 dtype=np.int32,
                 shuffle=True,
                 pad_position_value=512,
                 prefetch=1,
                 drop_remainder=True):
        self.files = input_files
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.max_mask_tokens = max_mask_tokens
        self.dtype = dtype
        self.file_index = 0
        self.data_index = 0
        self.shuffle = shuffle
        self.len = None
        self.pad_position_value = pad_position_value
        self.drop_remainder = drop_remainder
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        num_samples = pool.map(self.samples_in_file, self.files)
        pool.close()
        pool.join()
        self.total_samples = sum(num_samples)
        self.len = self.total_samples // (self.batch_size)
        self.num_prefetch_batches = prefetch
        self.prefetch_buffer = deque()
        if self.len < 1:
            raise ValueError(f"""Batch size {self.batch_size} larger than
                                number of samples in the TFRecord files {self.total_samples}.""")

        if self.len < self.num_prefetch_batches:
            raise ValueError(f"""Not enough samples to prefetch: (length = {self.len},
                            num_to_prefech = {self.num_prefetch_batches}),
                            lower the number of prefetch batches.""")
        self.samples_per_file = {f: n for (f, n) in
                                 zip(self.files, num_samples)}
        self.data = None
        self.counter = 0

    def samples_in_file(self, filename):
        reader = TfRecordReader(filename,
                                transforms={k: lambda x: x.numpy().astype(self.dtype)
                                            for k in KEYS})
        count = 0
        while reader.read_example():
            count += 1
        return count

    def __len__(self):
        return self.len

    def __iter__(self):
        self.file_index = 0
        self.data_index = 0
        self.counter = 0
        self.data = None
        if self.shuffle:
            random.shuffle(self.files)
        self.fill_buffer(self.num_prefetch_batches)
        return self

    def post_process(self, samples):
        batch_size, seq_len = samples['input_ids'].shape
        formatted_pos = self.pad_position_value * np.ones_like(samples['input_ids'])
        formatted_input = np.zeros_like(samples['input_ids'])
        formatted_seg = np.zeros_like(samples['segment_ids'])
        formatted_mask_labels = np.zeros((batch_size, self.max_mask_tokens),
                                         dtype=samples['masked_lm_ids'].dtype)

        valid_seq_positions = []
        valid_mask_positions = samples['masked_lm_weights'] == 1
        valid_mask_len = np.sum(valid_mask_positions, axis=1)
        for i, mask_pos in enumerate(samples['masked_lm_positions']):
            pos = [True] * seq_len
            for mask_index, m in enumerate(mask_pos):
                if mask_index < valid_mask_len[i]:
                    pos[m] = False
            valid_seq_positions.append(np.logical_and(pos, samples['input_ids'][i] != 0))
        valid_seq_len = np.minimum(np.sum(valid_seq_positions, axis=1) + self.max_mask_tokens,
                                   self.max_seq_length)
        unmasked_len = np.minimum(np.sum(valid_seq_positions, axis=1),
                                  self.max_seq_length - self.max_mask_tokens)
        for i in range(batch_size):
            target_mask_indices = np.arange(valid_mask_len[i])
            target_seq_indices = self.max_mask_tokens + np.arange(unmasked_len[i])
            source_mask_indices = samples['masked_lm_positions'][i][valid_mask_positions[i]]
            source_seq_indices = np.arange(seq_len)[valid_seq_positions[i]][:unmasked_len[i]]

            target_indices = np.hstack([target_mask_indices, target_seq_indices])
            source_indices = np.hstack([source_mask_indices, source_seq_indices])

            formatted_pos[i, target_indices] = source_indices
            formatted_input[i, target_indices] = samples['input_ids'][i, source_indices]
            formatted_seg[i, target_indices] = samples['segment_ids'][i, source_indices]
            formatted_mask_labels[i] = samples['masked_lm_ids'][i, :self.max_mask_tokens]

        return [formatted_input, formatted_pos, formatted_seg,
                valid_mask_len, valid_seq_len, formatted_mask_labels,
                samples['next_sentence_labels']]

    def __next__(self):
        if self.drop_remainder:
            if self.counter == self.len:
                raise StopIteration

        if len(self.prefetch_buffer) == 0:
            raise StopIteration

        result = self.prefetch_buffer.popleft()
        self.counter += 1
        self.fill_buffer(1)
        return result

    def fill_buffer(self, num_batches):
        if self.data is None:
            self.load_data()
        for _ in range(num_batches):
            curr_batch = []
            still_required = self.batch_size
            while still_required > 0:
                data = self.data[self.data_index:
                                 self.data_index + still_required]
                self.data_index += len(data)
                curr_batch += data
                still_required = self.batch_size - len(curr_batch)
                if still_required > 0:
                    if self.file_index < len(self.files):
                        self.load_data()
                    else:
                        break
            if len(curr_batch) == self.batch_size:
                result = {}
                for k in KEYS:
                    result[k] = np.vstack([item[k] for item in curr_batch])
                self.prefetch_buffer.append(self.post_process(result))

    def load_data(self):
        if self.file_index >= len(self.files):
            raise ValueError('No more files to load.')
        self.data = self.load_file(self.files[self.file_index])
        self.file_index += 1
        self.data_index = 0
        if self.shuffle:
            np.random.shuffle(self.data)

    def load_file(self, filename):
        reader = TfRecordReader(filename,
                                transforms={k: lambda x: x.numpy().astype(self.dtype)
                                            for k in KEYS})
        data = []
        ex = reader.read_example()
        while ex:
            data.append(ex)
            ex = reader.read_example()
        return data
