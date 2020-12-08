# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import os
from urllib import request
import tarfile
import subprocess
import tempfile


cifar10_data_dir = None


def download_cifar():
    """Download the CIFAR-10 dataset if it's not already available."""

    DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    dir_name = 'cifar-10-batches-bin'
    filename = "cifar-10-binary.tar.gz"
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "Datasets")
    filepath = os.path.join(data_dir, dir_name)

    if not os.path.exists(filepath):
        with tempfile.TemporaryDirectory(dir=data_dir) as tmpdirname:
            tmpfilepath = os.path.join(tmpdirname, filename)
            print('Downloading', filename, "to", tmpfilepath)
            tmpfilepath, _ = request.urlretrieve(DATA_URL, tmpfilepath)
            print('Successfully downloaded, extracting to', tmpdirname)
            tarfile.open(tmpfilepath, 'r:gz').extractall(tmpdirname)
            print('Moving', tmpdirname, "to", data_dir)
            try:
                os.rename(os.path.join(tmpdirname, dir_name), os.path.join(data_dir, dir_name))
            except OSError:
                pass

    return os.path.join(data_dir, dir_name)

cifar10_data_dir = download_cifar()


def run_train(**kwargs):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir('..')
    cmd = ['python3', 'train.py']
    args = [str(item) for sublist in kwargs.items() for item in sublist if item != '']
    cmd.extend(args)
    return subprocess.check_output(cmd).decode('utf-8')


def run_restore(mypath, **kwargs):
    cmd = ['python3', 'restore.py']
    args = [str(item) for sublist in kwargs.items() for item in sublist if item != '']
    cmd.extend(args)
    return subprocess.check_output(cmd, cwd=mypath).decode('utf-8')


def run_validation(mypath, **kwargs):
    cmd = ['python3', 'validation.py']
    args = [str(item) for sublist in kwargs.items() for item in sublist if item != '']
    cmd.extend(args)
    return subprocess.check_output(cmd, cwd=mypath).decode('utf-8')


def parse_csv(filepath):
    with open(filepath) as csv:
        lines = csv.read().split('\n')
        items = [line.split(',') for line in lines if line]
        results = {}  # dict with headers of csv as keys
        for i in range(len(items[0])):
            values = [float(v[i]) for v in items[1:]]
            results[items[0][i]] = values
    return results


def get_csv(out, name):
    log_dir = None
    for line in out.split('\n'):
        if line.find('Saving to ') != -1:
            log_dir = line[11:]
            break
    if not log_dir:
        raise ValueError("Couldn't find log directory from output")

    return parse_csv(os.path.join(log_dir, name))
