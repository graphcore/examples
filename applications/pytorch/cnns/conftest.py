# Copyright 2020 Graphcore Ltd.
import os
import subprocess

from efficientnet_pytorch import EfficientNet

from models.models import available_models


def download_images():
    # Download files required for some tests, only if not already downloaded
    cwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference")
    images_path = os.path.join(cwd, "images")
    if not os.path.exists(images_path):
        print("Getting images...")
        subprocess.check_output(["sh", "get_images.sh"], cwd=cwd)


def get_models():
    print("Getting models...")
    for model in available_models:
        if "efficientnet" in model:
            available_models[model].from_pretrained(model)
        else:
            available_models[model](pretrained=True)


def pytest_sessionstart(session):
    """Get the data required for the tests."""
    download_images()
    get_models()
