# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import subprocess
from pathlib import Path

from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet

from models.models import available_models


def download_images():
    # Download files required for some tests, only if not already downloaded
    cwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    images_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "images")
    if not os.path.exists(images_path):
        print("Getting images...")
        subprocess.check_output(["sh", "get_images.sh"], cwd=cwd)


def get_models():
    print("Getting models...")
    for model in available_models:
        available_models[model]["model"](pretrained=True)


def get_cifar10_dataset():
    print("Getting cifar10 dataset...")
    data_path = Path(__file__).parent.absolute().joinpath("data").joinpath("cifar10")
    datasets.CIFAR10(
        root=data_path.resolve(),
        train=True,
        download=True
    )
    datasets.CIFAR10(
        root=data_path.resolve(),
        train=False,
        download=True
    )


def pytest_sessionstart(session):
    """Get the data required for the tests."""
    get_cifar10_dataset()
    download_images()
    get_models()
