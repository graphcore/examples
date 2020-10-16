# Copyright 2020 Graphcore Ltd.
import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import sys
import logging
import poptorch


datasets_info = {"real": {"in": (3, 224, 224), "out": 1000},
                 "synthetic": {"in": (3, 224, 224), "out": 1000}}


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
        return self.images[index % len(self.images)]


class SynthDataset(Dataset):
    """
    A synthetic dataset.
    (no host->device IO), so this Dataset tries to minimize the data
    sent to the device. The data sent is used to build a tensor on the
    device side with the correct shape for inference.
    """
    def __init__(self, size=2000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return 0


def get_dataloader(batch_size, opts, num_iterations, synthetic=False):
    """
    A factory method to create a dataload responsible for sending data
    to the IPU device. This build the appropriate dataset, whether
    real or synthetic, and wraps it in a dataloader.
    """
    dataset_size = batch_size * \
        opts.device_iterations * \
        opts.replication_factor * \
        num_iterations

    if synthetic:
        dataset = SynthDataset(size=dataset_size)
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        dataset = SampleDataset(img_dir='./images', transform=transform, size=dataset_size)

    dataloader = poptorch.DataLoader(opts, dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return dataloader
