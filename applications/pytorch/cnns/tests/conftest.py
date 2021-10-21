# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
from pathlib import Path
import webdataset as wds
import json
import pytest
import gc
import torchvision
from torch.utils.data import DataLoader
import import_helper
from datasets import encode_sample, create_distributed_remaining
from utils import download_images, get_models, get_cifar10_dataset
from examples_tests.execute_once_per_fs import ExecuteOncePerFS


def convert_cifar2imagenet_format():
    print("Convert cifar10 dataset...")
    cifar_path = Path(__file__).parent.parent.absolute().joinpath("data").joinpath("cifar10")
    converted_path = Path(__file__).parent.parent.absolute().joinpath("data").joinpath("cifar10_raw").joinpath("validation")
    webdata_path = Path(__file__).parent.parent.absolute().joinpath("data").joinpath("cifar10_webdata")
    if os.path.exists(converted_path) and os.path.exists(webdata_path):
        return
    ds = torchvision.datasets.CIFAR10(
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
        train_ds = torchvision.datasets.CIFAR10(
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
