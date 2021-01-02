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
import multiprocessing
import os
from pathlib import Path
import sys
sys.path.append('..')
import models


datasets_info = {"real": {"out": 1000},
                 "synthetic": {"out": 1000},
                 "cifar10": {"out": 10},
                 "imagenet": {"out": 1000}}


def train_preprocess_steps(size=224):
    return [transforms.Resize(256),
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]


def validate_preprocess_steps(size=224):
    return [transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]


class ToHalf(torch.nn.Module):
    def forward(self, tensor):
        return tensor.half()


def get_preprocessing_pipeline(train, input_size=224, half_precision=False):
    if train:
        pipeline_steps = train_preprocess_steps(input_size[-1])
    else:
        pipeline_steps = validate_preprocess_steps(input_size[-1])
    if half_precision:
        pipeline_steps.append(ToHalf())
    return transforms.Compose(pipeline_steps)


class SynthDataset(Dataset):
    """
    A synthetic dataset.
    (no host->device IO), so this Dataset tries to minimize the data
    sent to the device. The data sent is used to build a tensor on the
    device side with the correct shape for inference.
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
    transform = get_preprocessing_pipeline(train, models.available_models[opts.model]["input_shape"], half_precision)
    # Determine the size of the small datasets
    if hasattr(opts, "iterations"):
        dataset_size = opts.batch_size * \
                       model_opts.device_iterations * \
                       model_opts.replication_factor * \
                       model_opts.Training.gradient_accumulation * \
                       opts.iterations

    # Select the right dataset
    if opts.data == "synthetic":
        if hasattr(opts, "iterations"):
            dataset = SynthDataset(models.available_models[opts.model]["input_shape"], size=dataset_size, half_precision=half_precision)
        else:
            dataset = SynthDataset(models.available_models[opts.model]["input_shape"], half_precision=half_precision)
    elif opts.data == "real":
        data_path = Path(__file__).parent.absolute().joinpath("images")
        if hasattr(opts, "iterations"):
            dataset = SampleDataset(img_dir=data_path, transform=transform, size=dataset_size)
        else:
            dataset = SampleDataset(img_dir=data_path, transform=transform)
    elif opts.data == "imagenet":
        if train:
            data_folder = 'train'
        else:
            data_folder = 'validation'
        dataset = torchvision.datasets.ImageFolder(os.path.join(opts.imagenet_data_path, data_folder), transform=transform)
    elif opts.data == "cifar10":
        data_path = Path(__file__).parent.absolute().joinpath("cifar10")
        dataset = torchvision.datasets.CIFAR10(root=data_path, train=train, download=True, transform=transform)

    num_loader_workers = min(32, multiprocessing.cpu_count())
    dataloader = poptorch.DataLoader(model_opts,
                                     dataset,
                                     batch_size=opts.batch_size,
                                     num_workers=num_loader_workers,
                                     shuffle=train,
                                     drop_last=True)
    if async_dataloader:
        return poptorch.AsynchronousDataAccessor(dataloader, load_indefinitely=True)
    else:
        return dataloader
