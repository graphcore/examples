# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from tqdm import tqdm
import os
import json
import argparse
import torchvision
import torch
from torchvision import transforms
import webdataset as wds
from PIL import Image

transform_img = [transforms.Resize(256)]


def get_args():
    parser = argparse.ArgumentParser(add_help=True, description='Convert ImageNet to WebDataset format.')
    parser.add_argument('--source', type=str, required=True, help='Path of the ImageNet dataset.')
    parser.add_argument('--target', type=str, required=True, help='Path of the converted dataset.')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset')
    parser.add_argument("--samples-per-shard", type=int, default=1024, help='Maximum number of samples in each chunks.')
    parser.add_argument('--format', choices=['img', 'tensor'], default='img', help="Determined the format of the saved images: jpegs or tensors")
    args = parser.parse_args()
    return args


def write_dataset(dataloader, target_path, chunksize):
    with wds.ShardWriter(target_path, maxcount=chunksize) as sink:
        for index, (data, label) in enumerate(tqdm(dataloader)):
            if isinstance(data, Image.Image):
                sample = {"__key__": str(index),
                          "jpg": data,
                          "cls": label}
            else:
                sample = {"__key__": str(index),
                          "pth": torch.tensor(data*255, dtype=torch.uint8),
                          "cls": label}
            sink.write(sample)

if __name__ == '__main__':
    opts = get_args()
    transform = transform_img
    if opts.format == "tensor":
        transform.append(transforms.ToTensor())

    if not os.path.exists(opts.target):
        os.mkdir(opts.target)

    # Train
    dataset_train = torchvision.datasets.ImageFolder(os.path.join(opts.source, "train"), transform=transforms.Compose(transform))
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=None, num_workers=16, shuffle=opts.shuffle)
    write_dataset(dataloader_train, os.path.join(opts.target, "train-%06d.tar"), opts.samples_per_shard)
    # Validation
    dataset_validation = torchvision.datasets.ImageFolder(os.path.join(opts.source, "validation"), transform=transforms.Compose(transform))
    dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=None, num_workers=16, shuffle=opts.shuffle)
    write_dataset(dataloader_validation, os.path.join(opts.target, "validation-%06d.tar"), opts.samples_per_shard)

    # Save metadatas of the dataset
    metadata = {"train_length": len(dataset_train),
                "validation_length": len(dataset_validation),
                "format": opts.format,
                "shuffle": opts.shuffle,
                "transform_pipeline": [str(step) for step in transform]}
    with open(os.path.join(opts.target, "metadata.json"), "w") as metafile:
        json.dump(metadata, metafile)
