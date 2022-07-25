# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

import glob
import multiprocessing

import numpy as np
import popdist
import torch
from tfrecord.reader import tfrecord_loader
from torch.utils.data import IterableDataset
from transformers import BertTokenizerFast

TFRECORD_KEYS = (  # Torch Model Keys
    'input_ids',  # input_ids                  : tokens after masking
    'input_mask',  # attention_mask             : 1 if padding token, 0 otherwise
    'segment_ids',  # token_type_ids             : sentence 0 or 1
    # masked_lm_positions        : position of masked tokens in input_ids
    'masked_lm_positions',
    # masked_lm_labels=None      : label of masked tokens with padding as 0.
    'masked_lm_ids',
    'next_sentence_labels'  # next_sentence_label=None   : 1 if next sentence, 0 otherwise
)


def expand_glob_files(files):
    result = []
    for filepath in files:
        expanded = glob.glob(filepath)
        if len(expanded) < 1:
            raise FileNotFoundError(f"Could not find file: {filepath}")
        result += expanded
    return result


class TFRecordPretrainingDataset(IterableDataset):
    """
    Preprocessed BERT pretraining dataset read from TFRecord files.

    Each datum is comprised of:
    - input_ids                  : tokens after masking
    - attention_mask             : 1 if padding token, 0 otherwise
    - token_type_ids             : sentence 0 or 1
    - masked_lm_positions        : position of masked tokens in input_ids
    - masked_lm_labels           : label of masked tokens with padding as 0
    - next_sentence_label        : 1 if next sentence, 0 otherwise

    Dataset internally loads `file_buffer_size` number of files into an
    internal buffer for shuffling. The first iteration of this dataset
    may take a few minutes while this internal buffer is loading.

    This Dataset is also compatible with multiprocessing. Each Dataloader worker
    will only read a shard of each TFRecord file, which will speed up the Dataloader
    and ensure no worker loads the same data as another worker.

    Parameters
    ----------
    files: List of TFRecord files containing the preprocessed pretraining data
    file_buffer_size: The number of files to read into the internal shuffle buffer
    shuffle: Shuffle the data?
    """

    def __init__(self,
                 input_files,
                 file_buffer_size=100,
                 shuffle=True):
        self.files = expand_glob_files(input_files)
        self.file_buffer_size = file_buffer_size
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        self.file_index = 0
        self.data_index = 0

    def samples_per_file(self, filename):
        reader = tfrecord_loader(filename,
                                 None,
                                 list(TFRECORD_KEYS))
        count = 0
        for _ in reader:
            count += 1
        return count

    def __len__(self):
        if getattr(self, "_len", None) is None:
            pool = multiprocessing.Pool(
                min(multiprocessing.cpu_count(), len(self.files)))
            num_samples = pool.map(self.samples_per_file, self.files)
            pool.close()
            pool.join()
            self._len = sum(num_samples)
        return self._len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            if popdist.isPopdistEnvSet():
                self.worker_id = worker_info.id + \
                    worker_info.num_workers * popdist.getInstanceIndex()
                self.shard = worker_info.id + worker_info.num_workers * \
                    popdist.getInstanceIndex(), worker_info.num_workers * popdist.getNumInstances()
            else:
                self.worker_id = worker_info.id
                self.shard = worker_info.id, worker_info.num_workers
        else:
            self.shard = None
        self.reset()
        if self.shuffle:
            np.random.shuffle(self.files)
        self.load_data()
        return self

    def __next__(self):
        if self.data_index >= len(self.data):
            self.load_data()
        data = self.data[self.data_index]
        self.data_index += 1
        return data

    def load_data(self):
        # This drops the remainder
        if self.file_index >= len(self.files):
            raise StopIteration
        self.data = []
        # Load multiple files into the data buffer at a time
        for _ in range(self.file_buffer_size):
            self.data += self.load_file()
            self.file_index += 1
            if self.file_index >= len(self.files):
                break
        if self.shuffle:
            np.random.shuffle(self.data)
        self.data_index = 0

    def load_file(self):
        reader = tfrecord_loader(self.files[self.file_index],
                                 self.files[self.file_index].replace(
                                     ".tfrecord", ".index"),
                                 list(TFRECORD_KEYS),
                                 self.shard)
        data = []
        for datum in reader:
            data.append(datum)
        return data


class WorkerInit:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, worker_id):
        np.random.seed((self.seed + worker_id) % np.iinfo(np.uint32).max)


def load_dataset(input_files):
    return TFRecordPretrainingDataset(input_files, shuffle=True, file_buffer_size=10)


if __name__ == "__main__":

    print("\nYou are executing bert_data directly.")
    print("Let's read the first input from sample dataset.")

    dataset = TFRecordPretrainingDataset(["data/sample_text.tfrecord"])
    print("dataset length: ", len(dataset), "\n")
    datum = next(iter(dataset))
    for (name, value) in datum.items():
        print(name, value.shape, value.dtype, type(value), value, "\n\n")

    print("And now, we are going to decode the tokens.\n")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased",
                                                  do_lower_case=True)
    print("\n\n", tokenizer.decode(datum['input_ids']), "\n\n")
