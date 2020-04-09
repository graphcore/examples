# Copyright 2020 Graphcore Ltd.
import numpy as np
import math


class Sampler(object):
    '''
    A Sampler can be used to provide indices that specify the order in which
    to iterate over the dataset. They are similar to PyTorch's Sampler.
    '''
    pass


class SequentialSampler(Sampler):
    '''
    Iterate over the data in the original order.
    '''
    def __init__(self, data_source):
        self.indices = list(range(len(data_source)))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class ShuffledSampler(Sampler):
    '''
    Iterate over the data in random order.
    '''
    def __init__(self, data_source, seed=0):
        self.num_samples = len(data_source)
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        indices = list(range(self.num_samples))
        self._rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedDataSampler(Sampler):
    '''
    Shard the dataset according to mpi_rank and mpi_size. Setting shuffle=True
    randomizes the indices. The data can be padded to be evenly divisible by
    the MPI size.
    '''
    def __init__(self,
                 data_source,
                 seed=0,
                 shuffle=True,
                 mpi_size=1,
                 mpi_rank=0,
                 padding=False):

        self.shuffle = shuffle
        self.mpi_size = mpi_size
        self.mpi_rank = mpi_rank
        self.data_source = data_source
        if padding:
            self.num_samples = int(math.ceil(len(data_source) * 1.0 / mpi_size))
        else:
            self.num_samples = len(data_source) // mpi_size

        self.total_samples = self.num_samples * self.mpi_size
        self.padding = padding

        self._rng = np.random.default_rng(seed)


    def __iter__(self):
        indices = list(range(len(self.data_source)))
        if self.shuffle:
            self._rng.shuffle(indices)

        if self.padding:
            # Pad indices to have evenly divisible number of samples for each replica
            indices += indices[:(self.total_samples - len(indices))]
            assert(len(indices) == self.total_samples)


        # Subsample data
        indices = indices[self.mpi_rank:self.total_samples:self.mpi_size]
        assert(len(indices) == self.num_samples)

        return iter(indices)

    def __len__(self):
        return self.num_samples
