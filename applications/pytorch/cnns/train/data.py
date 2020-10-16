# Copyright 2020 Graphcore Ltd.
import os
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from dataloader import ProcessPoolDataLoader, AsyncDataLoader
import poptorch
from functools import partial
import multiprocessing
import copy



datasets_info = {"cifar10": {"in": (3, 224, 224), "out": 10},
                 "imagenet": {"in": (3, 224, 224), "out": 1000},
                 "synthetic": {"in": (3, 224, 224), "out": 1000}}

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class SynthDataset(Dataset):
    """
    A synthetic dataset.
    (no host->device IO), so this Dataset tries to minimize the data
    sent to the device. The data sent is used to build a tensor on the
    device side with the correct shape for inference.
    """
    def __len__(self):
        return 60000

    def __getitem__(self, index):
        item = index % datasets_info["synthetic"]["out"]
        return item, item


def get_synthetic():
    dataset = SynthDataset()
    return dataset, dataset


def get_CIFAR10():

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
    return trainset, testset


def get_imagenet(data_path):
    trainset = torchvision.datasets.ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
    testset = torchvision.datasets.ImageFolder(os.path.join(data_path, 'validation'), transform=test_transform)
    return trainset, testset


def get_data(opts, model_opts):
    if opts.data == "cifar10":
        train_dataset, test_dataset = get_CIFAR10()
    elif opts.data == "imagenet":
        train_dataset, test_dataset = get_imagenet(opts.imagenet_data_path)
    elif opts.data == "synthetic":
        train_dataset, test_dataset = get_synthetic()
    else:
        raise Exception("The dataset is not supported")

    def half_precision(x): return x.half()

    def full_precision(x): return x.float()

    if opts.precision == "half":
        transform = half_precision
    elif opts.precision == "full":
        transform = full_precision

    num_loader_workers = min(32, multiprocessing.cpu_count())
    train_loader = partial(ProcessPoolDataLoader, model_opts, train_dataset, batch_size=opts.batch_size, shuffle=True, drop_last=True, num_workers=num_loader_workers)
    train_loader = AsyncDataLoader(train_loader, transform)
    inference_model_opts = poptorch.Options().deviceIterations(model_opts.device_iterations)
    inference_model_opts.replicationFactor(model_opts.replication_factor)
    test_loader = ProcessPoolDataLoader(inference_model_opts, test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=num_loader_workers)
    return train_loader, test_loader
