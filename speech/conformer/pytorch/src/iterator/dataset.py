# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2021 Mobvoi Inc. All Rights Reserved.
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
'''
This script has been adapted from some of the original WeNet repo found here:
[
    https://github.com/wenet-e2e/wenet/blob/main/wenet/dataset/dataset.py
]
Main changes:
    modified the dataset function and IPUCollateFn class
    wenet needs to return 5 features, and then here return 6 values, namely target_in and target_out.
'''


import random

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

import src.iterator.processor as processor
from src.utils.file_utils import read_lists


class Processor(IterableDataset):
    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        # TODO(Binbin Zhang): fix this
        # We can not handle uneven data for CV on DDP, so we don't
        # sample data by rank, that means every GPU gets the same
        # and all the CV data
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):
    def __init__(self, lists, shuffle=True, partition=True):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __len__(self):
        return len(self.lists)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        for index in indexes:
            # yield dict(src=src)
            data = dict(src=self.lists[index])
            data.update(sampler_info)
            yield data


def Dataset(data_type,
            data_list_file,
            symbol_table,
            conf,
            bpe_model=None,
            non_lang_syms=None,
            partition=True):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            bpe_model(str): model for english bpe part
            partition(bool): whether to do data partition in terms of rank
    """
    assert data_type in ['raw', 'shard']
    lists = read_lists(data_list_file)
    shuffle = conf.get('shuffle', False)
    dataset = DataList(lists, shuffle=shuffle, partition=partition)
    if data_type == 'shard':
        dataset = Processor(dataset, processor.url_opener)
        dataset = Processor(dataset, processor.tar_file_and_group)
    else:
        dataset = Processor(dataset, processor.parse_raw)

    dataset = Processor(dataset, processor.tokenize, symbol_table, bpe_model,
                        non_lang_syms, conf.get('split_with_space', False))

    resample_conf = conf.get('resample_conf', {})
    dataset = Processor(dataset, processor.resample, **resample_conf)

    speed_perturb = conf.get('speed_perturb', True)
    if speed_perturb:
        dataset = Processor(dataset, processor.speed_perturb)

    filter_conf = conf.get('filter_conf', {})
    dataset = Processor(dataset, processor.filter, **filter_conf)

    fbank_conf = conf.get('fbank_conf', {})
    dataset = Processor(dataset, processor.compute_fbank, **fbank_conf)

    spec_aug = conf.get('spec_aug', True)
    if spec_aug:
        spec_aug_conf = conf.get('spec_aug_conf', {})
        dataset = Processor(dataset, processor.spec_aug, **spec_aug_conf)

    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = Processor(dataset, processor.shuffle, **shuffle_conf)

    sort = conf.get('sort', True)
    if sort:
        sort_conf = conf.get('sort_conf', {})
        dataset = Processor(dataset, processor.sort, **sort_conf)

    batch_conf = conf.get('batch_conf', {})
    dataset = Processor(dataset, processor.batch, **batch_conf)
    dataset = Processor(dataset, processor.padding)
    return dataset


class IPUCollateFn:
    def __init__(self, max_feature_length, max_target_length, dtype, sos_id, eos_id):
        self.max_feature_length = max_feature_length
        self.max_target_length = max_target_length
        self.dtype = dtype
        self.sos = sos_id
        self.eos = eos_id

    def __call__(self, batch):
        batch_size = len(batch)
        feature_length = torch.cat([i[3] for i in batch])
        target_length = torch.cat([(i[4]) for i in batch])
        padded_feature = torch.zeros(
            size=[batch_size, self.max_feature_length, batch[0][1].size(2)])
        padded_target = torch.zeros(
            size=[batch_size, self.max_target_length], dtype=torch.long)
        keys = []
        for index, sample in enumerate(batch):
            padded_feature[index, :sample[1][0].size(0), :] = sample[1][0]
            padded_target[index, :sample[2][0].size(0)] = sample[2][0]
            keys.append(sample[0][0])
        keys = torch.as_tensor(keys)
        target_out = padded_target.scatter(
            1, target_length.unsqueeze(1).long(), self.eos)
        target_in = torch.nn.functional.pad(
            padded_target, (1, 0), mode='constant', value=self.sos
        )[:, :-1]
        if self.dtype == torch.float16:
            padded_feature = padded_feature.half()
        return keys, padded_feature, feature_length, target_in, target_out, target_length + 1
