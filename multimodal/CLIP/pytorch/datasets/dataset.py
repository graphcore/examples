# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 mlfoundations

# This file has been modified by Graphcore

import os
import random
from typing import List, Union

import numpy as np
import pandas as pd
import poptorch
import torch
from PIL import Image, ImageFile
from torchvision.transforms import (CenterCrop, Compose, Normalize,
                                    RandomResizedCrop, Resize, ToTensor)
from tqdm import tqdm

from .simple_tokenizer import SimpleTokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


ImageFile.LOAD_TRUNCATED_IMAGES = True


class GeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, length=10000, npx=224):
        self.length = length
        self.images = torch.randn(length, 3, npx, npx).half()
        self.input_ids = torch.randint(0, 49408, (length, 77))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.images[index], self.input_ids[index]


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, transforms, config):
        self.config = config
        self.image_filenames = image_filenames

        self.encoded_captions = captions
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {"input_ids": self.encoded_captions[idx]}

        image = Image.open(f"{self.config.image_path}/{self.image_filenames[idx]}")
        image = self.transforms(image)
        item['image'] = image.half()

        return item['image'], item['input_ids']

    def __len__(self):
        return len(self.image_filenames)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transforms(n_px=224, is_train=True):

    if is_train:
        return Compose([
            RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        return Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


def make_train_valid_dfs(config):
    image_names = []
    image_captions = []
    with open(config.captions_path, "r", encoding='utf-8') as f:
        all_lines = f.read().splitlines()
        print("reading the csv file...")
        for line in tqdm(all_lines):
            image_name, image_caption = line.split('\t')
            image_names.append(image_name)
            image_captions.append(image_caption)
    max_id = len(image_names)
    image_ids = np.arange(0, max_id)
    dataframe = pd.DataFrame({"id": image_ids, "image": image_names[: max_id], "caption": image_captions[: max_id]})

    return dataframe


class _WorkerInit:
    def __init__(self, seed, instance_id, worker_count):
        self.new_base_seed = seed + instance_id * worker_count

    def __call__(self, worker_id):
        seed = (self.new_base_seed + worker_id) % np.iinfo(np.uint32).max
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)


class DatasetRebatch:
    """
    Wrapper for DataLoader to hide multiple non-complete batches and combine them to full batches.
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


def build_loaders(config, opts, async_dataloader, return_remaining=False):
    rebatch_size = getattr(config, "dataloader_rebatch_size", None)
    rebatch_size = rebatch_size if rebatch_size is not None else min(1024, config.global_batch_size) // opts.Distributed.numProcesses
    rebatch_size = min(rebatch_size, config.global_batch_size)
    dataset_mode = poptorch.DataLoaderMode.AsyncRebatched if async_dataloader else poptorch.DataLoaderMode.Sync
    worker_initialization = _WorkerInit(config.random_seed, opts.Distributed.processId, config.dataloader_workers) if hasattr(config, 'random_seed') else None

    if config.host_generate_data or config.ipu_generate_data:
        train_dataset = GeneratedDataset()
    else:
        transforms = get_transforms(n_px=config.image_resolution)

        dataframe = make_train_valid_dfs(config)
        encoded_captions = tokenize(list(dataframe["caption"].values), context_length=config.context_length, truncate=config.truncate)
        image_values = dataframe["image"].values

        train_dataset = CLIPDataset(
            image_values,
            encoded_captions,
            transforms=transforms,
            config=config
        )

    train_dataloader = poptorch.DataLoader(opts,
                                           train_dataset,
                                           batch_size=config.batch_size,
                                           num_workers=config.dataloader_workers,
                                           shuffle=not (isinstance(train_dataset, torch.utils.data.IterableDataset)),
                                           drop_last=not (isinstance(train_dataset, torch.utils.data.IterableDataset)),
                                           persistent_workers=not (isinstance(train_dataset, torch.utils.data.IterableDataset)),
                                           auto_distributed_partitioning=not (isinstance(train_dataset, torch.utils.data.IterableDataset)),
                                           worker_init_fn=worker_initialization,
                                           mode=dataset_mode,
                                           rebatched_worker_size=rebatch_size,
                                           async_options={'load_indefinitely': True, "buffer_size": 8})

    if isinstance(train_dataset, torch.utils.data.IterableDataset):
        train_dataloader = DatasetRebatch(train_dataloader, config.global_batch_size, len(train_dataset), not(return_remaining))

    return train_dataloader


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    _tokenizer = SimpleTokenizer()
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
