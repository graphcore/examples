# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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
    def __init__(self, data_source, padding = 0):
        self.indices = list(range(len(data_source)))
        self.padding = padding
        if self.padding > 0:
            self.indices += self.indices[:self.padding]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class ShuffledSampler(Sampler):
    '''
    Iterate over the data in random order.
    '''
    def __init__(self, data_source, seed=0, padding=0):
        self.num_samples = len(data_source)
        self.padding = padding
        self.data_source = data_source
        self._rng = np.random.default_rng(seed)
        self.num_samples += self.padding

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        self._rng.shuffle(indices)
        if self.padding > 0:
            indices += indices[:self.padding]
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedDataSampler(Sampler):
    '''
    Shard the dataset according to popdist_rank and popdist_size. Setting shuffle=True
    randomizes the indices. The data can be padded to be evenly divisible by the popdist size.
    '''
    def __init__(self,
                 data_source,
                 seed=0,
                 shuffle=False,
                 popdist_size=1,
                 popdist_rank=0,
                 padding=False,
                 subsample_padding=0,
                 samples_per_step=1):

        self.shuffle = shuffle
        self.popdist_size = popdist_size
        self.popdist_rank = popdist_rank
        self.data_source = data_source
        self.subsample_padding = subsample_padding
        if padding:
            self.num_samples = int(math.ceil(len(data_source) * 1.0 / popdist_size))
        else:
            self.num_samples = len(data_source) // popdist_size
        if subsample_padding > 0:
            # Update padding size for no-drop-remainder given the new number of samples
            self.subsample_padding = int(np.ceil(self.num_samples / samples_per_step)) * samples_per_step - self.num_samples
        self.total_samples = self.num_samples * self.popdist_size
        self.padding = padding
        self._rng = np.random.default_rng(seed)
        self.returned_num_samples = self.num_samples + self.subsample_padding

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        if self.shuffle:
            self._rng.shuffle(indices)

        if self.padding:
            # Pad indices to have evenly divisible number of samples for each replica
            indices += indices[:(self.total_samples - len(indices))]
            assert(len(indices) == self.total_samples)

        # Subsample data
        indices = indices[self.popdist_rank:self.total_samples:self.popdist_size]
        assert(len(indices) == self.num_samples)

        if self.subsample_padding > 0:
            # Pad at the end AFTER the shuffling and the subsample, for no-drop-remainder
            indices += indices[:self.padding_sub]
        assert(len(indices) == self.returned_num_samples)

        return iter(indices)

    def __len__(self):
        return self.returned_num_samples


class SampleGenerator:
    def __init__(self, data_source, sampler=None):
        if sampler is None:
            sampler = SequentialSampler(data_source)
        self.data_source = data_source
        self.sampler = sampler

    def __iter__(self):
        self.index = 0
        self.data_iter = iter(self.data_source)
        self.index_iter = iter(self.sampler)
        return self

    def __next__(self):
        next_index = next(self.index_iter)
        data = next(self.data_iter)
        while self.index != next_index:
            data = next(self.data_iter)
            self.index += 1
        self.index += 1
        return data

    def __len__(self):
        return len(self.sampler)
