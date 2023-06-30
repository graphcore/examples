# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import glob
import torch
import popdist
import numpy as np
from torch.utils.data import Dataset, DistributedSampler, DataLoader
from typing import Iterator, Optional, Sized, Dict
import math


def expand_glob_files(files):
    result = []
    for filepath in files:
        expanded = glob.glob(filepath)
        if len(expanded) < 1:
            raise FileNotFoundError(f"Could not find file: {filepath}")
        result += expanded
    return result


class WorkerInit:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, worker_id):
        np.random.seed((self.seed + worker_id) % np.iinfo(np.uint32).max)


class DistributedSampler(torch.utils.data.DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_instances: Optional[int] = None,
        rank: Optional[int] = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        start_index: int = 0,
        epoch: int = 0,
    ) -> None:
        """
        Basically a torch DistributedSampler https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler
        working with popdist, and with a state so that training can be resumed.
        """
        if num_instances is None:
            if not popdist.isPopdistEnvSet():
                raise RuntimeError("Requires popdist to be available")
            num_instances = popdist.getNumInstances()
        if rank is None:
            if not popdist.isPopdistEnvSet():
                raise RuntimeError("Requires popdist to be available")
            rank = popdist.getInstanceIndex()
        if rank >= num_instances or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_instances - 1)
            )

        self.dataset = dataset
        self.num_replicas = num_instances
        self.rank = rank
        self.epoch = epoch
        self.drop_last = drop_last
        self.seed = seed
        self.start_index = start_index
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        indices = indices[self.start_index :]
        assert len(indices) == self.num_samples - self.start_index

        return iter(indices)

    def get_state(self) -> Dict:
        return {"start_index": self.start_index, "seed": self.seed, "epoch": self.epoch}

    def set_state(self, state: Dict) -> None:
        self.start_index = state["start_index"]
        self.seed = state["seed"]
        # equivalent to calling set_epoch
        self.epoch = state["epoch"]


class StatefulRandomSampler(torch.utils.data.RandomSampler):
    r"""Samples elements randomly.
    Args:
        data_source (Dataset): dataset to sample from
        start_index (int): index to start sampling from
        seed (int): seed for random number generator
        epoch (int): number of epochs. The generator is setup to draw samples deterministically based on
                     seed and epoch.
    """

    def __init__(self, data_source: Sized, start_index: int = 0, seed: int = 0, epoch: int = 0) -> None:
        self.data_source = data_source
        self.seed = seed
        self.start_index = start_index
        self.epoch = epoch

    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(self.num_samples, generator=generator).tolist()[self.start_index :]

    def get_state(self) -> Dict:
        return {"start_index": self.start_index, "seed": self.seed, "epoch": self.epoch}

    def set_state(self, state: Dict) -> None:
        self.start_index = state["start_index"]
        self.seed = state["seed"]
        self.epoch = state["epoch"]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class StatefulDataLoader(DataLoader):
    r"""DataLoader which keeps track of its own state and can be saved and resumed.
    Combines a dataset and a random sampler, and provides an iterable over
    the given dataset.

    The random sampler is created automatically. If popdist is available, a
    DistributedSampler is used.
    Otherwise, a StatefulRandomSampler.
    The dataloader automatically keeps track of the epochs so it's not
    necessary to set the epoch manually on the sampler.

    To deal with multiprocessing correctly, the index to resume the dataloader from
    has to be taken from the dataloader iterator, not from the random sampler.

    See :class:`~torch.utils.data.DataLoader` for init arguments.
    Don't use sampler and generator Args
    """

    def __init__(self, dataset, seed: int = 47, start_index: int = 0, epoch: int = 0, *args, **kwargs):
        if popdist.isPopdistEnvSet():
            sampler = DistributedSampler(dataset, shuffle=True, seed=seed, start_index=start_index, epoch=epoch)
        else:
            sampler = StatefulRandomSampler(dataset, seed=seed, start_index=start_index, epoch=epoch)

        DataLoader.__init__(self, dataset=dataset, sampler=sampler, *args, **kwargs)
        self.epochs = 0

    @property
    def last_index(self):
        if self._iterator:
            return self.sampler.start_index + self._iterator._num_yielded * self.batch_size
        else:
            return self.sampler.start_index

    def __iter__(self) -> "_BaseDataLoaderIter":
        # in the base dataloader implementation single process iterator
        # are recreated every time to avoid resetting its state
        # But we need to keep the state so we unify the behaviours
        if self._iterator is None:
            self._iterator = self._get_iterator()
        else:
            # happens on new epoch
            self.sampler.start_index = 0
            self.epochs += 1
            self.sampler.set_epoch(self.epochs)
            self._iterator._reset(self)
        return self._iterator

    def get_state(self) -> Dict:
        state = self.sampler.get_state()
        state["start_index"] = self.last_index
        return state

    def set_state(self, state: Dict) -> None:
        self.epochs = state["epoch"]
        self.sampler.set_state(state)

    def save(self, filename: str) -> None:
        torch.save(self.get_state(), filename)

    def resume(self, filename: str) -> None:
        state = torch.load(filename)
        self.set_state(state)
