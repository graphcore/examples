# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import random
import webdataset as wds
import torch
import json
import os
import braceexpand
import logging
from torch.utils.data import IterableDataset
from math import ceil
import import_helper
from datasets.create_webdataset import parse_transforms


def identity(x):
    return x


def split_datasets(dataset_filelist, ratio_in_first_set, worker_count=1):
    """
    Splits the dataset to two parts. The number of chunks in the first split is dividable with worker_count.
    """
    index_split = ceil(ratio_in_first_set*len(dataset_filelist))
    index_split = index_split - index_split % worker_count
    first_dataset = dataset_filelist[:index_split]
    second_dataset = dataset_filelist[index_split:]
    return first_dataset, second_dataset


def get_webdataset(args, opts, train=True, transform=identity, shuffle_buffer=5000, use_bbox_info=False, cache_ratio=0.0):
    subset_name = 'train' if train else 'validation'
    with open(os.path.join(args.imagenet_data_path, "metadata.json")) as metadata_file:
        metadata = json.load(metadata_file)
        dataset_size = metadata[f"{subset_name}_length"]
        data_format = metadata["format"]
        if train:
            done_preprocess = metadata["train_transform_pipeline"]
        else:
            done_preprocess = metadata["validation_transform_pipeline"]

    chunks = [file_name[-10:-4] for file_name in os.listdir(args.imagenet_data_path) if file_name.startswith(subset_name)]
    chunks.sort()  # sort the chunks so that they can be distributed properly
    # Handle missing batches in distributed case
    all_chunks = list(braceexpand.braceexpand(os.path.join(args.imagenet_data_path, subset_name + "-{" + chunks[0] + ".." + chunks[-1] + "}.tar")))
    all_chunks, _ = split_datasets(all_chunks, args.webdataset_percentage_to_use / 100.0)
    cache_ratio = getattr(args, 'webdataset_memory_cache_ratio', 0.0)
    memory_chunks, disk_chunks = split_datasets(all_chunks, cache_ratio, worker_count=opts.Distributed.numProcesses * args.dataloader_worker)
    if len(memory_chunks) > 0:
        memory_cache = MemoryCache(memory_chunks, data_format, opts, dataset_size * (1.0 - cache_ratio)//(opts.Distributed.numProcesses * args.dataloader_worker))
    else:
        memory_cache = None
    remaining_chunks = None
    if opts.Distributed.numProcesses > 1:
        distributed_path = os.path.join(args.imagenet_data_path, "distributed", f"{opts.Distributed.numProcesses}-instances")
        if os.path.exists(distributed_path):
            remaining_chunks = [os.path.join(distributed_path, f"{subset_name}-{i:06d}.tar") for i in range(opts.Distributed.numProcesses)]
        else:
            logging.warn("Data is not distributed correctly between instances, which may result in skipping samples. Recommended using distributed_webdataset.py before running.")
    node_splitting = DistributeNode(remaining_chunks, opts.Distributed.processId, opts.Distributed.numProcesses, seed=getattr(args, "seed", 0))
    data_length = dataset_size // opts.Distributed.numProcesses
    if dataset_size % opts.Distributed.numProcesses > opts.Distributed.processId:
        data_length += 1
    dataset = WebDataset(disk_chunks, length=data_length, shuffle_buffer=shuffle_buffer if train else 0, nodesplitter=node_splitting, memory_cache=memory_cache)

    # Remove transformations, which are already done during preprocess.
    transform = match_preprocess(transform, done_preprocess)
    dataset.dataset = decode_webdataset(dataset.dataset, data_format, transform, use_bbox_info=use_bbox_info, raw_data=False, memory_cache=memory_cache)
    return dataset


def decode_webdataset(dataset, data_format, transform, use_bbox_info=False, raw_data=False, memory_cache=None):
    if data_format == "img":
        dataset = dataset.decode("pil").to_tuple("jpg;png", "cls", "json")
    elif data_format == "tensor":
        dataset = dataset.decode("torch").to_tuple("pth", "cls", "json")
    else:
        raise Exception(f"Data format {data_format} is not supported.")
    if memory_cache is not None:
        dataset = wds.Processor(dataset, memory_cache.cache)
    if not raw_data:
        dataset = dataset.map(BboxInfoWrapper(transform, use_bbox_info))
    return dataset


class MemoryCache:
    def __init__(self, cached_urls, data_format, opts, dataset_length, cache_chunk_size=256):
        """
        Caches part of the dataset
        cached_urls: the chunks, which are stored in the memory
        data_format: img or tensor
        cache_chunk_size: number of samples in each cache block
        """
        self.cache_chunk_size = cache_chunk_size
        self.len_data = int(dataset_length)
        self.cached_urls = cached_urls
        node_distribute = DistributeNode(None, opts.Distributed.processId, opts.Distributed.numProcesses, remove_remaining_chunks=False)
        self.cache_dataset = WebDataset(cached_urls, length=0, shuffle_buffer=0, nodesplitter=node_distribute)
        self.cache_dataset.dataset = decode_webdataset(self.cache_dataset.dataset, data_format, transform=None, use_bbox_info=False, raw_data=True)
        self.cached_content = []
        self.is_new_iter = False

    def _load_cache(self):
        # Make sure each worker caches different chunks
        worker_info = torch.utils.data.get_worker_info()
        worker_id, num_workers = worker_info.id, worker_info.num_workers
        filtered_urls = [url for idx, url in enumerate(self.cached_urls) if idx % num_workers == worker_id]
        logging.info("Load dataset memory cache.")

        for sample in self.cache_dataset:
            self.cached_content.append(sample)

    def new_iter(self):
        """
        This method must be called at the beginning of each epoch
        """
        self.is_new_iter = True
        random.shuffle(self.cached_content)

    def cache(self, data):
        """
        The cache method: return samples from the memory first than return the samples from the disk.
        """

        if len(self.cached_content) == 0 and len(self.cached_urls) > 0:
            self._load_cache()
        # Return cached items at the beginning of the epoch
        if self.is_new_iter:
            self.is_new_iter = False
            cache_iter = iter(self.cached_content)
        else:
            cache_iter = None
        # Return other samples after
        # Add items from the cache after trigger_idx item
        trigger_idx = self.len_data // ((len(self.cached_content) // self.cache_chunk_size) + 1)
        for idx, sample in enumerate(data):
            if idx % trigger_idx == 0:
                try:
                    if cache_iter is not None:
                        for _ in range(self.cache_chunk_size):
                            yield next(cache_iter)
                except StopIteration:
                    cache_iter = None
            yield sample

        # return remaining cached samples
        if cache_iter is not None:
            try:
                while True:
                    yield next(cache_iter)
            except StopIteration:
                cache_iter = None


class BboxInfoWrapper():
    def __init__(self, transform, use_bbox_info):
        self.transform = transform
        self.use_bbox_info = use_bbox_info

    def __call__(self, sample):
        img, label, bbox = sample
        bbox = bbox["bbox"]
        if bbox is None or not self.use_bbox_info:
            img = self.transform(img)
        else:
            img = self.transform((img, bbox))
        return img, label


class DistributeNode:
    def __init__(self, remaining_chunks, instance_id, total_instance, seed=0, remove_remaining_chunks=True):
        """
        remove_remaining_chunks: if it is true the last chunks are not used (the samples must be among the remaining_chunks) as it may contain less samples as others
        """
        self.epoch = 0
        self.remaining_chunks = remaining_chunks
        self.instance_id = instance_id
        self.total_instance = total_instance
        self.base_seed = seed
        self.remove_remaining_chunks = remove_remaining_chunks

    def __call__(self, urls):
        # remove the chunks, which are already in the remaining_chunks
        if self.remove_remaining_chunks:
            remaining_chunks = len(urls) % self.total_instance
            if remaining_chunks == 0 and self.total_instance > 1:
                remaining_chunks = self.total_instance
            url_count = len(urls) - remaining_chunks
            urls = urls[:url_count]
        # reshuffle with the same seed in all instances
        random.Random(self.epoch + self.base_seed).shuffle(urls)
        self.epoch += 1
        start = (len(urls) // self.total_instance) * self.instance_id
        end = (len(urls) // self.total_instance) * (self.instance_id + 1)
        urls = urls[start:end]
        if self.remaining_chunks is not None:
            urls = urls + [self.remaining_chunks[self.instance_id]]
        return urls


class WebDataset(IterableDataset):
    def __init__(self, urls, length, shuffle_buffer, nodesplitter=None, memory_cache=None):
        super().__init__()
        self.memory_cache = memory_cache
        self.dataset = wds.WebDataset(urls, shardshuffle=True if shuffle_buffer > 1 else False, length=length, nodesplitter=nodesplitter)
        if shuffle_buffer > 1:
            self.dataset = self.dataset.shuffle(shuffle_buffer)

    def __iter__(self):
        if self.memory_cache is not None:
            self.memory_cache.new_iter()
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)


def match_preprocess(preprocess_pipeline, done_preprocess=[]):
    transforms = parse_transforms(done_preprocess)
    while len(transforms) > 0:
        if isinstance(transforms[0], type(preprocess_pipeline.transforms[0])) and vars(transforms[0]) == vars(preprocess_pipeline.transforms[0]):
            preprocess_pipeline.transforms.pop(0)
            transforms.pop(0)
        else:
            return preprocess_pipeline
    return preprocess_pipeline


class DatasetRebatch:
    """
    Wrapper for DataLoader to hide multiple non-complete batches and combine them to full batches
    """
    def __init__(self, dataloader, batch_size, total_samples, drop_last=True):
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.total_samples = total_samples
        self.drop_last = drop_last

    @property
    def combinedBatchSize(self):
        return self.batch_size

    def __iter__(self):
        self.remaining = None
        self.iterable_dataloader = iter(self.dataloader)
        self.end_iter = False
        return self

    def __len__(self):
        length = self.total_samples // self.batch_size
        if self.total_samples % self.batch_size > 0 and not self.drop_last:
            length += 1
        return length

    def __next__(self):
        if self.end_iter:
            raise StopIteration
        while True:
            try:
                tensor = next(self.iterable_dataloader)
            except StopIteration:
                self.end_iter = True
                if not(self.remaining is None) and self.remaining[0].size()[0] > 0 and not self.drop_last:
                    return self.remaining
                else:
                    raise StopIteration

            if tensor[0].size()[0] == self.batch_size:
                return tensor
            else:
                if self.remaining is None:
                    self.remaining = tensor
                else:
                    self.remaining = [torch.cat([buffer, current], dim=0) for buffer, current in zip(self.remaining, tensor)]
                    if self.remaining[0].size()[0] >= self.batch_size:
                        returning_tensor = [buffer[:self.batch_size] for buffer in self.remaining]
                        self.remaining = [buffer[self.batch_size:] for buffer in self.remaining]
                        return returning_tensor

    def terminate(self):
        self.dataloader.terminate()
