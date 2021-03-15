# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import glob
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import sys
import logging
import poptorch
import os
from pathlib import Path
import sys
sys.path.append('..')
import models
from datasets.webdataset_format import get_webdataset, DatasetRebatch
from datasets.preprocess import get_preprocessing_pipeline


datasets_info = {"real": {"out": 1000},
                 "synthetic": {"out": 1000},
                 "generated": {"out": 1000},
                 "cifar10": {"out": 10},
                 "imagenet": {"out": 1000}}


class GeneratedDataset(Dataset):
    """
    Generated dataset creates a random dataset with the given shape and precision.
    The size determines the number of items in the dataset.
    """
    def __init__(self, shape, size=60000, half_precision=False):
        self.size = size
        self.half_precision = half_precision
        self.data_shape = shape

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        synthetic_data = torch.rand(self.data_shape)
        if self.half_precision:
            synthetic_data = synthetic_data.half()
        return synthetic_data, index % datasets_info["synthetic"]["out"]


class SampleDataset(Dataset):
    """
    Sample dataset for inference to use in conjuntion with a
    DataLoader.
    """
    def __init__(self, img_dir, transform=None, size=2000):
        files = glob.glob("{}/*.jpg".format(img_dir))
        if len(files) == 0:
            logging.error('No images found. Run get_images.sh script. Aborting...')
            sys.exit()
        self.images = []
        for filename in files:
            img = Image.open(filename)
            if transform:
                img = transform(img)
            self.images.append(img)
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.images[index % len(self.images)], index % len(self.images)


def get_data(opts, model_opts, train=True, async_dataloader=False):
    """
    A factory method to create a dataload responsible for sending data
    to the IPU device. This build the appropriate dataset and wraps it in a dataloader.
    """
    if opts.precision[:3] == "16.":
        half_precision = True
    elif opts.precision[:3] == "32.":
        half_precision = False
    transform = get_preprocessing_pipeline(train, models.available_models[opts.model]["input_shape"][-1], half_precision, opts.normalization_location == "host")
    # Determine the size of the small datasets
    if hasattr(opts, "iterations"):
        dataset_size = opts.batch_size * \
                       model_opts.device_iterations * \
                       model_opts.replication_factor * \
                       model_opts.Training.gradient_accumulation * \
                       opts.iterations

    # Select the right dataset
    if opts.data in ["synthetic", "generated"]:
        if hasattr(opts, "iterations"):
            dataset = GeneratedDataset(models.available_models[opts.model]["input_shape"], size=dataset_size, half_precision=half_precision)
        else:
            dataset = GeneratedDataset(models.available_models[opts.model]["input_shape"], half_precision=half_precision)
    elif opts.data == "real":
        data_path = Path(__file__).parent.parent.absolute().joinpath("data").joinpath("images")
        if hasattr(opts, "iterations"):
            dataset = SampleDataset(img_dir=data_path, transform=transform, size=dataset_size)
        else:
            dataset = SampleDataset(img_dir=data_path, transform=transform)
    elif opts.data == "imagenet":
        if os.path.exists(os.path.join(opts.imagenet_data_path, 'metadata.json')):
            # WebDataset format
            dataset = get_webdataset(opts, model_opts, train, half_precision, transform=transform)
        else:
            # Original ImageNet format
            data_folder = 'train' if train else 'validation'
            dataset = torchvision.datasets.ImageFolder(os.path.join(opts.imagenet_data_path, data_folder), transform=transform)
    elif opts.data == "cifar10":
        data_path = Path(__file__).parent.parent.absolute().joinpath("data").joinpath("cifar10")
        dataset = torchvision.datasets.CIFAR10(root=data_path, train=train, download=True, transform=transform)
    mode = poptorch.DataLoaderMode.Async if async_dataloader and not isinstance(dataset, torch.utils.data.IterableDataset) else poptorch.DataLoaderMode.Sync
    dataloader = poptorch.DataLoader(model_opts,
                                     dataset,
                                     batch_size=opts.batch_size if not(isinstance(dataset, torch.utils.data.IterableDataset)) else None,
                                     num_workers=opts.dataloader_worker,
                                     shuffle=train and not(isinstance(dataset, torch.utils.data.IterableDataset)),
                                     drop_last=not(isinstance(dataset, torch.utils.data.IterableDataset)),
                                     persistent_workers = True,
                                     auto_distributed_partitioning = not isinstance(dataset, torch.utils.data.IterableDataset),
                                     worker_init_fn=None,
                                     mode=mode,
                                     async_options={'load_indefinitely': True})

    if isinstance(dataset, torch.utils.data.IterableDataset):
        global_batch_size = opts.batch_size * model_opts.device_iterations * model_opts.replication_factor * model_opts.Training.gradient_accumulation
        if async_dataloader:
            dataloader._accessor = poptorch.AsynchronousDataAccessor(DatasetRebatch(dataloader, global_batch_size), load_indefinitely=True)
        else:
            dataloader = DatasetRebatch(dataloader, global_batch_size)
    return dataloader
