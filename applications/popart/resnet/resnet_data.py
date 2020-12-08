# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import numpy as np
import random
import torch
import os
from torchvision import datasets, transforms
import resnet_dataloader


# Add the padding required. Do that in the data loader
class PadChannel(object):

    def __call__(self, pic):
        return torch.nn.functional.pad(pic, (0, 0, 0, 0, 0, 1), "constant", 0)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def load_dataset(opts, training=True):
    # For predictable input data
    torch.manual_seed(0)
    random.seed(0)

    if opts.dataset == "CIFAR-10":

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop([32, 32], padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                                (0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
            PadChannel()
        ])

        dataset = datasets.CIFAR10(root=opts.data_dir,
                                   train=training,
                                   transform=transform,
                                   download=True)

    elif opts.dataset == "IMAGENET":
        if training:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                    PadChannel()
                ])
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                    PadChannel()
                ])

        if training:
            imagedir = os.path.join(opts.data_dir, 'train')
        else:
            imagedir = os.path.join(opts.data_dir, 'validation')

        dataset = datasets.ImageFolder(root=imagedir,
                                       transform=transform)

    tensor_type = 'float16' if opts.precision == np.float16 else 'float32'

    # In the case there is not sufficient data in last batch drop it
    training_session_anchor_size = opts.batch_size*opts.batches_per_step*opts.gradient_accumulation_factor
    validation_session_anchor_size = opts.batch_size*opts.batches_per_step

    loader = resnet_dataloader.DataLoader(
        dataset,
        # data we need per replica
        batch_size=training_session_anchor_size if training else validation_session_anchor_size,
        tensor_type=tensor_type,
        shuffle=True, num_workers=opts.num_workers,
        drop_last=True)

    return DataSet(
        opts.batch_size,
        opts.batches_per_step,
        opts.samples_per_replica,
        opts.replication_factor,
        opts.gradient_accumulation_factor,
        loader,
        training,
        opts.precision)


class DataSet:

    def __init__(self, batch_size, batches_per_step, samples_per_replica,
                 replication_factor, gradient_accumulation_factor, loader, training=True, dtype=np.float16):
        self.training = training
        self.dtype = dtype
        self.loader = loader
        self.num_examples = len(loader) * batch_size * batches_per_step
        self.batch_size = batch_size
        self.samples_per_replica = samples_per_replica
        self.replication_factor = replication_factor
        self.gradient_accumulation_factor = gradient_accumulation_factor
        self.batches_per_step = min(batches_per_step,
                                    self.num_examples //
                                    self.samples_per_replica)
        self.inputs_per_step = self.batch_size * self.batches_per_step

        self.steps_per_epoch = self.num_examples // self.inputs_per_step

        # This needs to be done here as the DataLoader will fork the workers.
        # Fork does not work well once the program has started
        self.loader_iterator = self.loader.__iter__()

    def __iter__(self):
        self.loader_iterator.reset()
        return self

    def __len__(self):
        return self.steps_per_epoch

    def __next__(self):

        # Get the next image/label
        images, labels = next(self.loader_iterator)

        # Determine the shape of the batch based on batch size, gradient accumulation
        # and replication
        batch_shape = [self.samples_per_replica]

        if self.replication_factor > 1:
            batch_shape = [self.replication_factor] + batch_shape

        if self.training:
            if self.gradient_accumulation_factor > 1:
                batch_shape = [self.gradient_accumulation_factor] + batch_shape

        if self.batches_per_step > 1:
            batch_shape = [self.batches_per_step] + batch_shape

        # Reshape the input
        images = images.reshape(batch_shape + list(images.shape[1:]))
        labels = labels.reshape(batch_shape)

        return images, labels
