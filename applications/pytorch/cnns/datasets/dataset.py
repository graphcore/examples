# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import glob
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import sys
import logging
import poptorch
import os
import random
from pathlib import Path
import import_helper
import models
from datasets.preprocess import get_preprocessing_pipeline
from datasets.raw_imagenet import ImageNetDataset


datasets_info = {
    "real": {"out": 1000},
    "synthetic": {"out": 1000},
    "generated": {"out": 1000},
    "cifar10": {"out": 10},
    "imagenet": {"out": 1000},
}


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
        torch.manual_seed(0)
        synthetic_data = torch.randint(0, 255, self.data_shape)
        if self.eightbit:
            synthetic_data = synthetic_data.byte()
        else:
            if self.half_precision:
                synthetic_data = synthetic_data.half()
            else:
                synthetic_data = synthetic_data.float()
        self.data = synthetic_data

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data, index % datasets_info["synthetic"]["out"]


class SampleDataset(Dataset):
    """
    Sample dataset for inference to use in conjuntion with a
    DataLoader.
    """
    def __init__(self, img_dir, transform=None, size=2000):
        files = glob.glob("{}/*.jpg".format(img_dir))
        if len(files) == 0:
            logging.error('No images found. Run get_images.sh script. Aborting...')
            sys.exit(1)
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


def get_data(args, opts, train=True, async_dataloader=False, return_remaining=False, fine_tuning=False):
    """
    A factory method to create a dataload responsible for sending data
    to the IPU device. This build the appropriate dataset and wraps it in a dataloader.
    """
    logging.info("Loading the data")
    input_shape = models.model_input_shape(args, train)
    if args.precision[:3] == "16.":
        half_precision = True
    elif args.precision[:3] == "32.":
        half_precision = False
    use_bbox_info = getattr(args, "use_bbox_info", False)

    if args.data in ["real", "imagenet", "cifar10"]:
        transform = get_preprocessing_pipeline(train, input_shape[-1],
                                               half_precision, args.normalization_location == "host", eightbit = args.eight_bit_io,
                                               use_bbox_info=use_bbox_info, fine_tuning=fine_tuning)
    # Determine the size of the small datasets
    if hasattr(args, "iterations"):
        dataset_size = args.batch_size * \
                       opts.device_iterations * \
                       opts.replication_factor * \
                       opts.Training.gradient_accumulation * \
                       args.iterations

    # Select the right dataset
    if args.data in ["synthetic", "generated"]:
        if hasattr(args, "iterations"):
            dataset = GeneratedDataset(input_shape, size=dataset_size, half_precision=half_precision, eightbit=args.eight_bit_io)
        else:
            dataset = GeneratedDataset(input_shape, half_precision=half_precision, eightbit=args.eight_bit_io)
    elif args.data == "real":
        data_path = Path(__file__).parent.parent.absolute().joinpath("data").joinpath("images")
        if hasattr(args, "iterations"):
            dataset = SampleDataset(img_dir=data_path, transform=transform, size=dataset_size)
        else:
            dataset = SampleDataset(img_dir=data_path, transform=transform)
    elif args.data == "imagenet":
        assert os.path.exists(args.imagenet_data_path), f"{args.imagenet_data_path} does not exist!"
        # Original ImageNet format
        data_folder = 'train' if train else 'validation'
        data_folder = os.path.join(args.imagenet_data_path, data_folder)
        bboxes = os.path.join(args.imagenet_data_path, 'imagenet_2012_bounding_boxes.csv') if use_bbox_info and train else None   # use bboxes only for training
        dataset = ImageNetDataset(data_folder, transform=transform, bbox_file=bboxes)
    elif args.data == "cifar10":
        data_path = Path(__file__).parent.parent.absolute().joinpath("data").joinpath("cifar10")
        dataset = torchvision.datasets.CIFAR10(root=data_path, train=train, download=True, transform=transform)
    global_batch_size = args.batch_size * opts.device_iterations * opts.replication_factor * opts.Training.gradient_accumulation
    if async_dataloader:
        if global_batch_size == 1:
            # Avoid rebatch overhead
            mode = poptorch.DataLoaderMode.Async
        else:
            mode = poptorch.DataLoaderMode.AsyncRebatched
    else:
        mode = poptorch.DataLoaderMode.Sync
    worker_initialization = _WorkerInit(args.seed, opts.Distributed.processId, args.dataloader_worker) if hasattr(args, 'seed') else None
    rebatch_size = getattr(args, "dataloader_rebatch_size", None)
    rebatch_size = rebatch_size if rebatch_size is not None else min(1024, global_batch_size) // opts.Distributed.numProcesses
    # Make sure rebatch size is smaller than global batch size
    rebatch_size = min(rebatch_size, global_batch_size)
    dataloader = poptorch.DataLoader(opts,
                                     dataset,
                                     batch_size=args.batch_size,
                                     num_workers=args.dataloader_worker,
                                     shuffle=train,
                                     drop_last= not(return_remaining),
                                     persistent_workers = True,
                                     auto_distributed_partitioning = True,
                                     worker_init_fn=worker_initialization,
                                     mode=mode,
                                     rebatched_worker_size=rebatch_size,
                                     async_options={'load_indefinitely': True})
    return dataloader


class _WorkerInit:
    def __init__(self, seed, instance_id, worker_count):
        self.new_base_seed = seed + instance_id * worker_count

    def __call__(self, worker_id):
        seed = (self.new_base_seed + worker_id) % np.iinfo(np.uint32).max
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
