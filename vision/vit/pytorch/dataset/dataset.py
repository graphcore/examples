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

import json
import os
import random
from functools import partial, wraps
from pathlib import Path

import poptorch
import torch
import torchvision
from poptorch import DataLoader
from torch.utils.data import Dataset, IterableDataset

from dataset.preprocess import get_preprocessing_pipeline


def get_random_datum(config):
    dataset = GeneratedDataset(
        shape=[3, 224, 224], size=config.samples_per_step, half_precision=True, byteio=config.byteio
    )
    return dataset.stacked_batch()


class GeneratedDataset(Dataset):
    """
    Generated dataset creates a random dataset with the given shape and precision.
    The size determines the number of items in the dataset.
    """

    def __init__(self, shape, size=4096, half_precision=False, byteio=False):
        self.data_shape = shape
        self.default_generate_size = 1024
        self.half_precision = half_precision
        self.byteio = byteio
        self.size = size
        self.images, self.labels = self.generate()
        self.samples = list(range(size))  # fake list of (path, class).

    def generate(self):
        size_ = min(self.size, self.default_generate_size)
        images_ = torch.rand((size_, *self.data_shape))
        labels_ = torch.randint(0, 2, [size_], dtype=torch.long)
        if self.byteio:
            images_ = images_.byte()
        elif self.half_precision:
            images_ = images_.half()
        return images_, labels_

    def stacked_batch(self):
        ret_images = []
        ret_labels = []
        repeat_size = self.size // self.default_generate_size
        left_over = self.size % self.default_generate_size
        for i in range(repeat_size):
            ret_images.append(self.images)
            ret_labels.append(self.labels)
        ret_images.append(self.images[:left_over])
        ret_labels.append(self.labels[:left_over])
        return torch.vstack(ret_images), torch.cat(ret_labels)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index >= self.size or index < 0:
            raise IndexError
        else:
            index_ = index % self.default_generate_size
        return self.images[index_], self.labels[index_]


def cache_output(filename):
    """
    Dump the output of a function to a cache file in json format
    """

    def decorator(f):
        @wraps(f)
        def func(self, directory, *args, **kwargs):
            filepath = Path.cwd() / (Path(directory).stem + filename)
            if filepath.is_file():
                out = json.loads(filepath.read_text())
            else:
                out = f(self, directory, *args, **kwargs)
                filepath.write_text(json.dumps(out))
            return out

        return func

    return decorator


class ImageNetDataset(torchvision.datasets.ImageFolder):
    @cache_output(filename="_classes.json")
    def find_classes(self, *args, **kwargs):
        classes = super().find_classes(*args, **kwargs)
        return classes

    @cache_output(filename="_structure.json")
    def make_dataset(self, *args, **kwargs):
        dataset = super().make_dataset(*args, **kwargs)
        return dataset


def get_data(config, model_opts, train=True, async_dataloader=False):
    dataset = get_dataset(config, model_opts, train=train)
    dataloader = get_dataloader(config, model_opts, dataset, train=train, async_dataloader=async_dataloader)
    return dataloader


def get_dataset(config, model_opts, train=True):
    """
    A factory method to create a dataloader responsible for sending data
    to the IPU device. This build the appropriate dataset and wraps it in a dataloader.
    """
    if config.precision.startswith("16."):
        half_precision = True
    elif config.precision.startswith("32."):
        half_precision = False

    # is byteio is true, normalization should be performed on the device
    config.normalize_image_on_device = False
    if config.byteio:
        config.normalize_image_on_device = True

    transform = get_preprocessing_pipeline(
        train,
        224,
        half_precision,
        normalize=not config.normalize_image_on_device,
        extra_aug=config.extra_aug,
        byteio=config.byteio,
    )

    # Determine the size of the small datasets
    dataset_size = (
        config.micro_batch_size
        * model_opts.device_iterations
        * config.replication_factor
        * model_opts.Training.gradient_accumulation
        * config.iterations
    )

    # Select the right dataset
    if config.dataset == "generated":
        dataset = GeneratedDataset(
            (3, 224, 224), size=dataset_size, half_precision=half_precision, byteio=config.byteio
        )

    elif config.dataset in ["imagenet1k", "imagenet21k"]:
        dataset = ImageNetDataset(
            os.path.join(config.dataset_path, "train" if train else "validation"), transform=transform
        )

    elif config.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=config.dataset_path, train=train, download=True, transform=transform
        )
    else:
        raise Exception("Dataset type not recognized: %s" % config.dataset)
    return dataset


def get_dataloader(config, model_opts, dataset, train=True, async_dataloader=False):
    if async_dataloader:
        if config.global_batch_size == 1:
            # avoid rebatch overhead
            mode = poptorch.DataLoaderMode.Async
        else:
            mode = poptorch.DataLoaderMode.AsyncRebatched
            if config.rebatched_worker_size is not None:
                config.rebatched_worker_size = min(config.rebatched_worker_size, config.global_batch_size)
    else:
        mode = poptorch.DataLoaderMode.Sync

    async_options = {
        "sharing_strategy": poptorch.SharingStrategy.SharedMemory,
        "load_indefinitely": True,
        "early_preload": True,
        "miss_sleep_time_in_ms": 0,
        "buffer_size": 8,
    }

    if train and not (isinstance(dataset, IterableDataset)):
        shuffle = True
    else:
        shuffle = False

    if not (isinstance(dataset, IterableDataset)):
        batch_size = config.micro_batch_size
    else:
        batch_size = None

    dataloader = poptorch.DataLoader(
        model_opts,
        dataset,
        batch_size=batch_size,
        num_workers=config.dataloader_workers,
        shuffle=shuffle,
        drop_last=not (isinstance(dataset, IterableDataset)),
        persistent_workers=True,
        auto_distributed_partitioning=not isinstance(dataset, IterableDataset),
        worker_init_fn=None,
        mode=mode,
        async_options=async_options,
        rebatched_worker_size=config.rebatched_worker_size,
        collate_fn=None,
    )
    return dataloader
