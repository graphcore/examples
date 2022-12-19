# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
from typing import Optional

import horovod.tensorflow as hvd
import popdist
import tensorflow as tf
import tensorflow_datasets as tfds
from mpi4py import MPI

from . import abstract_dataset


class TFDSDataset(abstract_dataset.AbstractDataset):

    def __init__(self,
                 dataset_name: str,
                 dataset_path: str,
                 split: str,
                 shuffle: bool = True,
                 deterministic: bool = False,
                 seed: Optional[int] = None):

        if dataset_path is not None:
            if not os.path.exists(dataset_path):
                raise ValueError(f'Directory {dataset_path} does not exist')

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.split = split
        self.shuffle = shuffle
        self.deterministic = deterministic
        self.seed = seed

    def read_single_image(self) -> tf.data.Dataset:
        read_config = tfds.ReadConfig(try_autocache=False,
                                      skip_prefetch=True,
                                      shuffle_seed=self.seed)
        if self.deterministic:
            read_config.interleave_cycle_length = 1
            read_config.interleave_block_length = 1
            read_config.num_parallel_calls_for_decode = 1
            read_config.num_parallel_calls_for_interleave_files = 1

        if popdist.getNumInstances() > 1:
            # This is to allow downloading from 1 local host only
            if hvd.local_rank() == 0:
                ds, info_ds = tfds.load(self.dataset_name,
                                        data_dir=self.dataset_path,
                                        split=self.split,
                                        download=True,
                                        as_supervised=True,
                                        with_info=True,
                                        shuffle_files=self.shuffle,
                                        read_config=read_config)
            # This barrier forces the other instances to wait until the local root instance has downloaded the data
            MPI.COMM_WORLD.Barrier()
            if hvd.local_rank() != 0:
                ds, info_ds = tfds.load(self.dataset_name,
                                        data_dir=self.dataset_path,
                                        split=self.split,
                                        download=False,
                                        as_supervised=True,
                                        with_info=True,
                                        shuffle_files=self.shuffle,
                                        read_config=read_config)
        else:
            ds, info_ds = tfds.load(self.dataset_name,
                                    data_dir=self.dataset_path,
                                    split=self.split,
                                    as_supervised=True,
                                    with_info=True,
                                    shuffle_files=self.shuffle,
                                    read_config=read_config)

        num_examples = info_ds.splits[self.split].num_examples

        if popdist.getNumInstances() > 1:
            ds = ds.shard(num_shards=popdist.getNumInstances(), index=popdist.getInstanceIndex())

        ds = ds.cache()

        if self.split == 'train' and self.shuffle:
            ds = ds.shuffle(buffer_size=num_examples // popdist.getNumInstances(), seed=self.seed)

        return ds
