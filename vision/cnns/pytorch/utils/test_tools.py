# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import subprocess
import sys
from pathlib import Path
from torchvision import datasets
from models.factory import available_models


def run_script(script_name, parameters, python=True):
    """
    Run a script and returns with the result
    The script runs from root folder (cnns)
    """
    # Select root directory
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_list = parameters.split(" ")
    if python:
        cmd = [get_current_interpreter_executable(), script_name]
    else:
        cmd = [script_name]
    cmd = cmd + param_list
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.PIPE).decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(f"TEST FAILED")
        print(f"stdout={e.stdout.decode('utf-8',errors='ignore')}")
        print(f"stderr={e.stderr.decode('utf-8',errors='ignore')}")
        raise
    return out


def get_current_interpreter_executable():
    if sys.executable is not None and (sys.executable.endswith("python") or sys.executable.endswith("python3")):
        return sys.executable
    else:
        print(
            "Unable to get current interpreter executable, falling back to 'python'. "
            "This may cause issues with incompatible python environment."
        )
    return "python"


def get_test_accuracy(output):
    prefix = "Accuracy on test set:"
    pos_start = output.rfind(prefix)
    pos_end = pos_start + output[pos_start:].find("%")
    return float(output[pos_start + len(prefix) + 1 : pos_end])


def get_train_accuracy(output):
    prefix = "accuracy:"
    pos_start = output.rfind(prefix)
    pos_end = pos_start + output[pos_start:].find("%")
    return float(output[pos_start + len(prefix) + 1 : pos_end])


def get_max_thoughput(output):
    if output.find("throughput:") != -1:
        pos = output.find("max:")
        return float(output[pos + 4 :].split(",")[0][:-7])
    else:
        return -1


def download_images():
    """Download files required for some tests, only if not already downloaded"""

    cwd = Path(__file__).parent.parent.absolute().joinpath("datasets")
    images_path = Path(__file__).parent.parent.absolute().joinpath("data").joinpath("images")
    if not os.path.exists(images_path):
        print("Getting images...")
        subprocess.check_output(["sh", "get_images.sh"], cwd=cwd)


def get_models():
    print("Getting models...")
    for model in available_models:
        available_models[model]["model"](pretrained=True)


def get_cifar10_dataset():
    print("Getting cifar10 dataset...")
    data_path = Path(__file__).parent.parent.absolute().joinpath("data").joinpath("cifar10")
    datasets.CIFAR10(root=data_path.resolve(), train=True, download=True)
    datasets.CIFAR10(root=data_path.resolve(), train=False, download=True)


def install_turbo_jpeg():
    print("Installing turbojpeg...")
    try:
        cwd = Path(__file__).parent.parent.absolute()
        subprocess.run(["make", "install-turbojpeg"], cwd=cwd, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"TEST FAILED")
        print(f"stdout={e.stdout.decode('utf-8',errors='ignore')}")
        print(f"stderr={e.stderr.decode('utf-8',errors='ignore')}")
        raise
