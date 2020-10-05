# Copyright 2020 Graphcore Ltd.
import glob
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import sys
import logging

datasets_shape = {"real": {"in": (3, 224, 224), "out": 1000},
                  "synthetic": {"in": (3, 224, 224), "out": 1000}}


class SampleDataset(Dataset):
    """
    Sample dataset for inference to use in conjuntion with a
    DataLoader.
    """
    def __init__(self, img_dir, transform=None):
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

    def __len__(self):
        return 2000

    def __getitem__(self, index):
        return self.images[index % len(self.images)]


class SynthDataset(Dataset):
    """
    A "synthetic" dataset. Poptorch does not support synthetic
    (no host->device IO), so this Dataset tries to minimize the data
    sent to the device. The data sent is used to build a tensor on the
    device side with the correct shape for inference.
    """
    def __init__(self):
        pass

    def __len__(self):
        return 2000

    def __getitem__(self, index):
        return 0


def get_dataloader(batch_size, synthetic=False):
    """
    A factory method to create a dataload responsible for sending data
    to the IPU device. This build the appropriate dataset, whether
    real or synthetic, and wraps it in a dataloader.
    """
    if synthetic:
        dataset = SynthDataset()
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        dataset = SampleDataset(img_dir='./images', transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return dataloader
