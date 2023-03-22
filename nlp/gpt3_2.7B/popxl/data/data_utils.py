# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import glob
import torch
import popdist
import multiprocessing
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, IterableDataset
from tfrecord.reader import tfrecord_loader

TFRECORD_KEYS = ["input_ids"]  # Torch Model Keys


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
    Preprocessed GPT2 pretraining dataset read from TFRecord files.
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
    """

    def __init__(self, input_files, shuffle=True):
        self.files = expand_glob_files(input_files)
        self.shuffle = shuffle
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
            pool = multiprocessing.Pool(min(multiprocessing.cpu_count(), len(self.files)))
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
                self.shard = (
                    worker_info.id + worker_info.num_workers * popdist.getInstanceIndex(),
                    worker_info.num_workers * popdist.getNumInstances(),
                )
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
            self.reader = tfrecord_loader(
                self.files[self.file_index],
                self.files[self.file_index].replace(".tfrecord", ".index"),
                list(TFRECORD_KEYS),
                self.shard,
            )
            self.file_index += 1
            datum = next(self.reader)
        input_ids = torch.tensor(datum[TFRECORD_KEYS[0]], dtype=torch.long)
        return input_ids


class MyDataset(Dataset):
    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        input_ids = self.input_list[index]
        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def __len__(self):
        return len(self.input_list)


def load_dataset(input_files):
    """
    load train and valid dataset
    """
    dataset = TFRecordPretrainingDataset(input_files, shuffle=True)
    return dataset


def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    data = {"input_ids": input_ids[:, :-1], "labels": labels[:, 1:]}
    return data


class WorkerInit:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, worker_id):
        np.random.seed((self.seed + worker_id) % np.iinfo(np.uint32).max)
