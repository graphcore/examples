# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import random
import webdataset as wds
import torch
import json
import os
import braceexpand
import logging
from torchvision import transforms
from torch.utils.data import IterableDataset
from math import ceil
import sys
sys.path.append('..')
from datasets.create_webdataset import parse_transforms



def identity(x):
    return x


def truncate_datasets(dataset_filelist, percentage_to_keep):
    index_to_delete_from = ceil(percentage_to_keep*len(dataset_filelist)/100)
    dataset_filelist = dataset_filelist[:index_to_delete_from]
    return dataset_filelist


def get_webdataset(opts, model_opts, train=True, transform=identity, shuffle_buffer=5000, use_bbox_info=False):
    subset_name = 'train' if train else 'validation'
    with open(os.path.join(opts.imagenet_data_path, "metadata.json")) as metadata_file:
        metadata = json.load(metadata_file)
        dataset_size = metadata[f"{subset_name}_length"]
        data_format = metadata["format"]
        if train:
            done_preprocess = metadata["train_transform_pipeline"]
        else:
            done_preprocess = metadata["validation_transform_pipeline"]

    chunks = [file_name[-10:-4] for file_name in os.listdir(opts.imagenet_data_path) if file_name.startswith(subset_name)]
    chunks.sort()  # sort the chunks so that they can be distributed properly
    # Handle missing batches in distributed case
    all_chunks = list(braceexpand.braceexpand(os.path.join(opts.imagenet_data_path, subset_name + "-{" + chunks[0] + ".." + chunks[-1] + "}.tar")))
    all_chunks = truncate_datasets(all_chunks, opts.webdataset_percentage_to_use)
    remaining_chunks = None
    if model_opts.Distributed.numProcesses > 1:
        distributed_path = os.path.join(opts.imagenet_data_path, "distributed", f"{model_opts.Distributed.numProcesses}-instances")
        if os.path.exists(distributed_path):
            remaining_chunks = [os.path.join(distributed_path, f"{subset_name}-{i:06d}.tar") for i in range(model_opts.Distributed.numProcesses)]
        else:
            logging.warn("Data is not distributed correctly between instances, which may result in skipping samples. Recommended using distributed_webdataset.py before running.")
    node_splitting = DistributeNode(remaining_chunks, model_opts.Distributed.processId, model_opts.Distributed.numProcesses, seed=getattr(opts, "seed", 0))
    data_length = dataset_size // model_opts.Distributed.numProcesses
    if dataset_size % model_opts.Distributed.numProcesses > model_opts.Distributed.processId:
        data_length += 1
    dataset = WebDataset(all_chunks, length=data_length, shuffle_buffer=shuffle_buffer if train else 0, nodesplitter=node_splitting)

    # Remove transformations, which are already done during preprocess.
    transform = match_preprocess(transform, done_preprocess)
    dataset.dataset = decode_webdataset(dataset.dataset, data_format, transform, use_bbox_info=use_bbox_info)
    return dataset


def decode_webdataset(dataset, data_format, transform, use_bbox_info=False):
    if data_format == "img":
        dataset = dataset.decode("pil").to_tuple("jpg;png", "cls", "json").map(BboxInfoWrapper(transform, use_bbox_info))
    elif data_format == "tensor":
        dataset = dataset.decode("torch").to_tuple("pth", "cls", "json").map(BboxInfoWrapper(transform, use_bbox_info))
    else:
        raise Exception(f"Data format {data_format} is not supported.")

    return dataset


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
    def __init__(self, remaining_chunks, instance_id, total_instance, seed=0):
        self.epoch = 0
        self.remaining_chunks = remaining_chunks
        self.instance_id = instance_id
        self.total_instance = total_instance
        self.base_seed = seed

    def __call__(self, urls):
        # remove the chunks, which are already in the remaining_chunks
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
    def __init__(self, urls, length, shuffle_buffer, nodesplitter=None):
        super().__init__()
        self.dataset = wds.WebDataset(urls, shardshuffle=True if shuffle_buffer > 1 else False, length=length, nodesplitter=nodesplitter)
        if shuffle_buffer > 1:
            self.dataset = self.dataset.shuffle(shuffle_buffer)

    def __iter__(self):
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
