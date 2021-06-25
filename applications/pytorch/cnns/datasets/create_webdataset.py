# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from tqdm import tqdm
import random
import re
import os
import json
import argparse
import torchvision
import torch
from checksumdir import dirhash
from torchvision import transforms
import webdataset as wds
from PIL import Image
import sys
sys.path.append('..')
from datasets.raw_imagenet import ImageNetDataset

available_transforms = [('Resize\([0-9]+\)', transforms.Resize),
                        ('CenterCrop\([0-9]+\)', transforms.CenterCrop)]


def parse_transforms(transforms):
    preprocess_steps = []
    for step in transforms:
        matched = False
        for pattern, process in available_transforms:
            if re.match(pattern, step):
                parameter = int(re.search('[0-9]+', step)[0])
                preprocess_steps.append(process(parameter))
                matched = True
                break
        assert matched, f'Could not interpret {step} transformation'
    return preprocess_steps


def get_args():
    parser = argparse.ArgumentParser(add_help=True, description='Convert ImageNet to WebDataset format.')
    parser.add_argument('--source', type=str, required=True, help='Path of the ImageNet dataset.')
    parser.add_argument('--target', type=str, required=True, help='Path of the converted dataset.')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset')
    parser.add_argument('--seed', type=int, help='Seed of the shuffle')
    parser.add_argument('--train-preprocess-steps', type=str, nargs='+', default=[], help='Provide the preprocessing steps for training. Options: [Resize(<size>), CenterCrop(<size>)')
    parser.add_argument('--validation-preprocess-steps', type=str, nargs='+', default=[], help='Provide the preprocessing steps for validation. Options: [Resize(<size>), CenterCrop(<size>)')
    parser.add_argument("--samples-per-shard", type=int, default=1024, help='Maximum number of samples in each chunks.')
    parser.add_argument('--format', choices=['img', 'tensor'], default='img', help="Determined the format of the saved images: jpegs or tensors")
    args = parser.parse_args()
    return args


def encode_sample(data, label, index, bbox=None):
    if isinstance(data, Image.Image):
        sample = {"__key__": str(index),
                  "jpg": data,
                  "cls": label,
                  "json": {"bbox": bbox}}
    else:
        sample = {"__key__": str(index),
                  "pth": torch.tensor(data*255, dtype=torch.uint8),
                  "cls": label,
                  "json": {"bbox": bbox}}
    return sample


def write_dataset(dataloader, target_path, chunksize, transform=None):
    with wds.ShardWriter(target_path, maxcount=chunksize) as sink:
        for index, (data, label) in enumerate(tqdm(dataloader)):
            if isinstance(data, tuple) or isinstance(data, list):
                bbox = data[1]
                data = data[0]
            else:
                bbox = None
            if transform is not None:
                data = transform(data)
            sink.write(encode_sample(data, label, index, bbox=bbox))

if __name__ == '__main__':
    opts = get_args()
    train_preprocess = parse_transforms(opts.train_preprocess_steps)
    validation_preprocess = parse_transforms(opts.validation_preprocess_steps)
    if opts.format == "tensor":
        train_preprocess.append(transforms.ToTensor())
        validation_preprocess.append(transforms.ToTensor())

    if not os.path.exists(opts.target):
        os.mkdir(opts.target)

    # Train
    if opts.seed is not None:
        torch.manual_seed(opts.seed)
    dataset_train = ImageNetDataset(os.path.join(opts.source, "train"), bbox_file=os.path.join(opts.source, 'imagenet_2012_bounding_boxes.csv'))
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=None, num_workers=16, shuffle=opts.shuffle)
    write_dataset(dataloader_train, os.path.join(opts.target, "train-%06d.tar"), opts.samples_per_shard, transform=transforms.Compose(train_preprocess))
    # Validation
    if opts.seed is None:
        opts.seed = seed = random.randint(0, 2**32-1)
    torch.manual_seed(opts.seed)
    dataset_validation = ImageNetDataset(os.path.join(opts.source, "validation"), bbox_file=None)
    dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=None, num_workers=16, shuffle=opts.shuffle)
    write_dataset(dataloader_validation, os.path.join(opts.target, "validation-%06d.tar"), opts.samples_per_shard, transform=transforms.Compose(validation_preprocess))
    checksum = dirhash(opts.target)
    # Save metadatas of the dataset
    metadata = {"train_length": len(dataset_train),
                "validation_length": len(dataset_validation),
                "format": opts.format,
                "shuffle": opts.shuffle,
                "train_transform_pipeline": opts.train_preprocess_steps,
                "validation_transform_pipeline": opts.validation_preprocess_steps,
                "seed": opts.seed,
                "checksum": checksum}
    with open(os.path.join(opts.target, "metadata.json"), "w") as metafile:
        json.dump(metadata, metafile)
