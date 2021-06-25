# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import glob
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import sys
import logging
import poptorch
import os
import random
from pathlib import Path
import sys
sys.path.append('..')
import models
from datasets.webdataset_format import get_webdataset, DatasetRebatch
from datasets.tfrecord_format import get_tfrecord
from datasets.preprocess import get_preprocessing_pipeline
from datasets.raw_imagenet import ImageNetDataset


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
    def __init__(self, shape, size=60000, half_precision=False, eightbit=False):
        self.size = size
        self.half_precision = half_precision
        self.eightbit = eightbit
        self.data_shape = shape

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        torch.manual_seed(0)
        synthetic_data = torch.randint(0, 255, self.data_shape)
        if self.eightbit:
            synthetic_data = synthetic_data.byte()
        else:
            if self.half_precision:
                synthetic_data = synthetic_data.half()
            else:
                synthetic_data = synthetic_data.float()
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


def get_data(opts, model_opts, train=True, async_dataloader=False, return_remaining=False):
    """
    A factory method to create a dataload responsible for sending data
    to the IPU device. This build the appropriate dataset and wraps it in a dataloader.
    """
    if opts.precision[:3] == "16.":
        half_precision = True
    elif opts.precision[:3] == "32.":
        half_precision = False
    use_bbox_info = getattr(opts, "use_bbox_info", False)

    if opts.data in ["real", "imagenet", "cifar10"]:
        transform = get_preprocessing_pipeline(train, models.available_models[opts.model]["input_shape"][-1],
                                               half_precision, opts.normalization_location == "host", eightbit = opts.eight_bit_io,
                                               use_bbox_info=use_bbox_info)
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
            dataset = GeneratedDataset(models.available_models[opts.model]["input_shape"], size=dataset_size, half_precision=half_precision, eightbit=opts.eight_bit_io)
        else:
            dataset = GeneratedDataset(models.available_models[opts.model]["input_shape"], half_precision=half_precision, eightbit=opts.eight_bit_io)
    elif opts.data == "real":
        data_path = Path(__file__).parent.parent.absolute().joinpath("data").joinpath("images")
        if hasattr(opts, "iterations"):
            dataset = SampleDataset(img_dir=data_path, transform=transform, size=dataset_size)
        else:
            dataset = SampleDataset(img_dir=data_path, transform=transform)
    elif opts.data == "imagenet":
        assert os.path.exists(opts.imagenet_data_path), f"{opts.imagenet_data_path} does not exist!"
        if os.path.exists(os.path.join(opts.imagenet_data_path, 'metadata.json')):
            # WebDataset format
            dataset = get_webdataset(opts, model_opts, train, transform=transform, use_bbox_info=use_bbox_info)
        else:
            data_folder = 'train' if train else 'validation'
            data_folder = os.path.join(opts.imagenet_data_path, data_folder)
            if os.path.exists(data_folder):
                # Original ImageNet format
                bboxes = os.path.join(opts.imagenet_data_path, 'imagenet_2012_bounding_boxes.csv') if use_bbox_info and train else None   # use bboxes only for training
                dataset = ImageNetDataset(data_folder, transform=transform, bbox_file=bboxes)
            else:
                # TFRecord format
                dataset = get_tfrecord(opts, model_opts, train, transform=transform, use_bbox_info=use_bbox_info)
    elif opts.data == "cifar10":
        data_path = Path(__file__).parent.parent.absolute().joinpath("data").joinpath("cifar10")
        dataset = torchvision.datasets.CIFAR10(root=data_path, train=train, download=True, transform=transform)
    global_batch_size = opts.batch_size * model_opts.device_iterations * model_opts.replication_factor * model_opts.Training.gradient_accumulation
    if async_dataloader:
        if global_batch_size == 1:
            # Avoid rebatch overhead
            mode = poptorch.DataLoaderMode.Async
        else:
            mode = poptorch.DataLoaderMode.AsyncRebatched
    else:
        mode = poptorch.DataLoaderMode.Sync
    worker_initialization = _WorkerInit(opts.seed, model_opts.Distributed.processId, opts.dataloader_worker) if hasattr(opts, 'seed') else None
    rebatch_size = getattr(opts, "dataloader_rebatch_size", None)
    rebatch_size = rebatch_size if rebatch_size is not None else min(1024, global_batch_size) // model_opts.Distributed.numProcesses
    dataloader = poptorch.DataLoader(model_opts,
                                     dataset,
                                     batch_size=opts.batch_size,
                                     num_workers=opts.dataloader_worker,
                                     shuffle=train and not(isinstance(dataset, torch.utils.data.IterableDataset)),
                                     drop_last= not(return_remaining) and not isinstance(dataset, torch.utils.data.IterableDataset),
                                     persistent_workers = True,
                                     auto_distributed_partitioning = not isinstance(dataset, torch.utils.data.IterableDataset),
                                     worker_init_fn=worker_initialization,
                                     mode=mode,
                                     rebatched_worker_size=rebatch_size,
                                     async_options={'load_indefinitely': True})
    if isinstance(dataset, torch.utils.data.IterableDataset):
        len_dataset = len(dataset)
        if not isinstance(dataset, torch.utils.data.IterableDataset) and model_opts.Distributed.numProcesses > 1:
            len_dataset = len(dataset) // model_opts.Distributed.numProcesses
            if len(dataset) % model_opts.Distributed.numProcesses > model_opts.Distributed.processId:
                len_dataset += 1
        dataloader = DatasetRebatch(dataloader, global_batch_size, len_dataset, not(return_remaining))
    return dataloader


class _WorkerInit:
    def __init__(self, seed, instance_id, worker_count):
        self.new_base_seed = seed + instance_id * worker_count

    def __call__(self, worker_id):
        seed = (self.new_base_seed + worker_id) % np.iinfo(np.uint32).max
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
