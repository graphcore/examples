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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset
from poptorch import DataLoader
from poptorch.enums import DataLoaderMode
import popdist
from transformers import BertTokenizerFast
from tfrecord.reader import tfrecord_loader


TFRECORD_KEYS = (           # Torch Model Keys
    'input_ids',            # input_ids                  : tokens after masking
    'input_mask',           # attention_mask             : 0 if padded token, 1 otherwise
    'segment_ids',          # token_type_ids             : sentence 0 or 1
    'masked_lm_positions',  # masked_lm_positions        : position of masked tokens in input_ids
    'masked_lm_ids',        # masked_lm_labels=None      : label of masked tokens with padding as 0
    'next_sentence_labels'  # next_sentence_label=None   : 1 if next sentence, 0 otherwise
)

TFRECORD_KEYS_PACKED = (
    'packed_input_ids',             # : tokens after masking
    'packed_input_mask',            # : 0 if padded token. 1, 2 or 3 if token belongs to 1st, 2nd or 3rd sequence resp.
    'packed_segment_ids',           # : sentence 0 or 1 for each sequence in the pack
    'packed_position_ids',          # : position of tokens relative to each sequence
    'packed_masked_lm_positions',   # : absolute position of masked tokens in input_ids
    'packed_masked_lm_ids',         # : label of masked tokens with padding as 0
    'packed_masked_lm_mask',     # : 0 if padded token. 1, 2 or 3 if masked token belongs to 1st, 2nd or 3rd sequence resp.
    'packed_next_sentence_labels',  # : 1 if next sentence, 0 otherwise
    'packed_next_sentence_mask'  # : 1 if sequence is present in pack, 0 if not
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

    This Dataset is compatible with multiprocessing. Each Dataloader worker
    will only read a shard of each TFRecord file, which will speed up the Dataloader
    and ensure no worker loads the same data as another worker. You are strongly
    advised to use a large number (e.g. 64) of dataloader workers because firstly,
    more workers could support high throughput, and secondly, more workers could
    give us more stochasticity and thus better convergence.


    Parameters
    ----------
    files: List of TFRecord files containing the preprocessed pretraining data
    shuffle: Shuffle the data?
    packed_data: Use packed data?
    """
    def __init__(self,
                 input_files,
                 shuffle=True,
                 packed_data=False):
        self.files = expand_glob_files(input_files)
        self.shuffle = shuffle
        if packed_data:
            self.tfrecord_keys = TFRECORD_KEYS_PACKED
        else:
            self.tfrecord_keys = TFRECORD_KEYS
        self.reset()

    def reset(self):
        self.file_index = 0
        self.reader = iter([])

    def samples_per_file(self, filename):
        index_filename = filename.replace(".tfrecord", ".index")
        count = sum(1 for _ in open(index_filename))
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
                self.worker_id = worker_info.id + worker_info.num_workers * popdist.getInstanceIndex()
                self.shard = worker_info.id + worker_info.num_workers * popdist.getInstanceIndex(), worker_info.num_workers * popdist.getNumInstances()
            else:
                self.worker_id = worker_info.id
                self.shard = worker_info.id, worker_info.num_workers
        else:
            self.shard = None
        self.reset()
        if self.shuffle:
            np.random.shuffle(self.files)
        return self

    def __next__(self):
        try:
            datum = next(self.reader)
        except StopIteration:
            if self.file_index >= len(self.files):
                raise StopIteration
            self.reader = tfrecord_loader(self.files[self.file_index],
                                          self.files[self.file_index].replace(".tfrecord", ".index"),
                                          list(self.tfrecord_keys),
                                          self.shard)
            self.file_index += 1
            datum = next(self.reader)
        datum = [datum[key] for key in self.tfrecord_keys]
        return datum


class GeneratedPretrainingDataset(Dataset):
    """
    Dataset that randomly generates mock BERT pretraining data.

    Each datum is comprised of:
    - input_ids                  : tokens after masking
    - attention_mask             : 1 if padding token, 0 otherwise
    - token_type_ids             : sentence 0 or 1
    - masked_lm_positions        : position of masked tokens in input_ids
    - masked_lm_labels           : label of masked tokens with padding as 0
    - next_sentence_label        : 1 if next sentence, 0 otherwise

    Parameters
    ----------
    vocab_size: BERT vocabulary size
    sequence_length: Sequence length
    mask_tokens: the number of mask tokens
    length: Length of generated dataset
    seed: Random seed
    packed_data: Use packed data?
    """
    def __init__(self, vocab_size, sequence_length, mask_tokens, length=1, seed=42, packed_data=False):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.mask_tokens = mask_tokens
        self.length = length
        self.seed = seed
        self.packed_data = packed_data
        self.data = self.generate_data()

    def generate_data(self):
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            input_ids = torch.randint(0, self.vocab_size,
                                      [self.sequence_length],
                                      dtype=torch.long)
            input_mask = torch.ones_like(input_ids)
            segment_ids = torch.zeros_like(input_ids)
            masked_lm_positions = torch.randint(0, self.sequence_length,
                                                [self.mask_tokens],
                                                dtype=torch.long)
            masked_lm_ids = torch.randint(0, self.vocab_size,
                                          [self.mask_tokens],
                                          dtype=torch.long)
            if self.packed_data:
                packed_position_ids = torch.arange(self.sequence_length)
                packed_masked_lm_mask = torch.ones_like(masked_lm_ids, dtype=torch.float)
                packed_next_sentence_labels = torch.randint(0, 2, [3], dtype=torch.long)
                packed_next_sentence_mask = torch.ones_like(packed_next_sentence_labels, dtype=torch.float)
                return input_ids, input_mask, segment_ids, packed_position_ids, masked_lm_positions, masked_lm_ids, packed_masked_lm_mask, packed_next_sentence_labels, packed_next_sentence_mask
            else:
                next_sentence_labels = torch.randint(0, 2, [1], dtype=torch.long)
                return input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels

    def __len__(self):
        return self.length

    def __getitem__(self, __):
        return self.data


def get_generated_datum(config):
    result = []
    dataset = GeneratedPretrainingDataset(config.vocab_size,
                                          config.sequence_length,
                                          config.mask_tokens,
                                          packed_data=config.packed_data)
    data = (dataset[i] for i in range(config.samples_per_step))
    for batches in zip(*data):
        result.append(torch.stack(batches))
    return result


class _WorkerInit:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, worker_id):
        np.random.seed((self.seed + worker_id) % np.iinfo(np.uint32).max)


def get_dataloader(config, opts):
    if config.dataset == 'generated':
        dataset = GeneratedPretrainingDataset(config.vocab_size,
                                              config.sequence_length,
                                              config.mask_tokens,
                                              config.samples_per_step,
                                              config.random_seed,
                                              packed_data=config.packed_data)
    elif config.dataset == 'pretraining':
        dataset = TFRecordPretrainingDataset(config.input_files, packed_data=config.packed_data)
    else:
        raise RuntimeError(f"Unknown dataset '{config.dataset}', aborting.")

    loader = DataLoader(opts,
                        dataset,
                        batch_size=config.micro_batch_size,
                        num_workers=config.dataloader_workers,
                        worker_init_fn=_WorkerInit(config.random_seed),
                        auto_distributed_partitioning = not isinstance(dataset, torch.utils.data.IterableDataset),
                        mode=DataLoaderMode.AsyncRebatched if config.async_dataloader else DataLoaderMode.Sync)
    return loader


if __name__ == "__main__":

    print("\nYou are executing bert_data directly.")
    print("Let's read the first input from sample dataset.")

    dataset = TFRecordPretrainingDataset(["data/sample_text.tfrecord"])
    print("dataset length: ", len(dataset), "\n")
    first = next(iter(dataset))
    named_datum = zip(['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids', 'next_sentence_labels'], first)
    for (name, value) in iter(named_datum):
        print(name, value.shape, value.dtype, type(value), value, "\n\n")

    print("And now, we are going to decode the tokens.\n")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased",
                                                  do_lower_case=True)
    print("\n\n", tokenizer.decode(first[0]), "\n\n")
