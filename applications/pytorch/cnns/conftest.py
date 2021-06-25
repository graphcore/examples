# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import subprocess
from pathlib import Path
import webdataset as wds
import json
import subprocess
import pytest
import functools
import gc

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from models.models import available_models
from datasets import encode_sample, create_distributed_remaining


class ExecuteOncePerFS:
    """Adds synchronization to the execution of a function so it only executes
    once per file-system."""

    def __init__(self, lockfile, file_list, timeout, retries=10):
        self.lockfile = lockfile
        self.file_list = file_list
        self.timeout = timeout
        self.retries = retries

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            # Race to become master process
            result = None
            try:
                with open(self.lockfile, "x"):
                    # Master process executes function
                    result = fn(*args, **kwargs)
            except FileExistsError:
                pass

            # Every process waits for files to be created
            attempts = 0
            sleep_time = self.timeout/self.retries
            remaining_files = self.file_list[:]
            while attempts < self.retries:
                remaining_files = [
                    path for path in remaining_files if not os.path.exists(path)]
                if len(remaining_files) == 0:
                    return result

                time.sleep(sleep_time)
                attempts += 1

            # If we are here it means that we timed out...
            raise RuntimeError(f"Timed out waiting for {remaining_files} to be made.")
        return wrapped


def download_images():
    """Download files required for some tests, only if not already downloaded"""

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


def convert_cifar2imagenet_format():
    print("Convert cifar10 dataset...")
    cifar_path = Path(__file__).parent.absolute().joinpath("data").joinpath("cifar10")
    converted_path = Path(__file__).parent.absolute().joinpath("data").joinpath("cifar10_raw").joinpath("validation")
    webdata_path = Path(__file__).parent.absolute().joinpath("data").joinpath("cifar10_webdata")
    if os.path.exists(converted_path) and os.path.exists(webdata_path):
        return
    ds = datasets.CIFAR10(
            root=cifar_path.resolve(),
            train=False,
            download=True
    )
    dl = DataLoader(ds, batch_size=None)

    # Create raw imagenet format
    if not os.path.exists(converted_path):
        os.makedirs(converted_path)
        for idx, (img, label) in enumerate(dl):
            image_folder = os.path.join(converted_path, str(label))
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            img.save(os.path.join(converted_path, str(label), "{}.jpg".format(idx)))

    # Create webdataset format
    if not os.path.exists(webdata_path):
        os.makedirs(webdata_path)
        with wds.ShardWriter(str(webdata_path.joinpath("validation-%06d.tar")), maxcount=256) as sink:
            for index, (data, label) in enumerate(dl):
                sink.write(encode_sample(data, label, index))

        # Train dataset
        train_ds = datasets.CIFAR10(
                    root=cifar_path.resolve(),
                    train=True,
                    download=True
                    )
        train_dl = DataLoader(train_ds, batch_size=None)
        with wds.ShardWriter(str(webdata_path.joinpath("train-%06d.tar")), maxcount=256) as sink:
            for index, (data, label) in enumerate(train_dl):
                sink.write(encode_sample(data, label, index))
        metadata = {"format": "img", "validation_length": 10000, "train_length": 60000, "validation_transform_pipeline": [], "train_transform_pipeline": []}
        with open(os.path.join(webdata_path, "metadata.json"), "w") as metafile:
            json.dump(metadata, metafile)

        # Create distributed chunks for 4 node
        class HelperClass:
            def __init__(self):
                pass
        opts = HelperClass()
        opts.num_instances = 4
        opts.target = webdata_path
        create_distributed_remaining(opts, "validation")


@pytest.fixture(autouse=True)
def cleanup():
    # Explicitly clean up to make sure we detach from the IPU and
    # free the graph before the next test starts.
    gc.collect()


@ExecuteOncePerFS(lockfile="lockfile.lock", file_list=[], timeout=120, retries=20)
def init_tests():
    """Get the data required for the tests."""
    get_cifar10_dataset()
    convert_cifar2imagenet_format()
    download_images()
    get_models()


def pytest_sessionstart(session):
    init_tests()
    os.environ["POPTORCH_WAIT_FOR_IPU"] = "1"
