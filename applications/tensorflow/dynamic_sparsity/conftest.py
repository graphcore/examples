# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path
from subprocess import run, PIPE

import os
import tempfile
import time
import shutil
import subprocess

import pytest

public_examples_dir = Path(__file__).parent.parent.parent.parent
build_dir = Path(__file__).parent


@pytest.fixture
def ipu_sparse_ops(scope="session"):
    """This function builds the ipu_sparse_ops
    library for any tests that rely on it.
    """
    build_path = Path(
        public_examples_dir,
        "applications",
        "tensorflow",
        "dynamic_sparsity"
    )
    completed = run(['python-config', '--extension-suffix'], stdout=PIPE)
    extension = completed.stdout.decode().replace('\n', '')
    shared_libs = [f'host_utils{extension}', 'libsparse_matmul.so']
    paths = [Path(build_path, "ipu_sparse_ops", f) for f in shared_libs]

    # Use exclusive lockfile to avoid race conditions on the build:
    lock_path = Path(build_path, ".ipu_sparse_ops.pytest.build.lockfile")
    try:
        with open(lock_path, "x") as lockfile:
            print("\nCleaning dynamic_sparsity")
            run(['make', 'clean'], cwd=build_path)
            print("\nBuilding dynamic_sparsity")
            run(['make', '-j'], cwd=build_path)
    except FileExistsError as e:
        print("\nipu_sparse_ops is already building.")

    exist = [path.exists() for path in paths]
    timeout = 15
    while not all(exist):
        time.sleep(1)
        exist = [path.exists() for path in paths]
        timeout -= 1
        if timeout == 0:
            raise RuntimeError("Timeout waiting for ipu_sparse_ops to build.")


@pytest.fixture
def wikitext_103_dataset(scope="session"):
    """Checks that wikitext-103 is available and if not downloads it."""
    # Lookup dataset storage location
    dataset_dir = os.getenv(
        'CI_GPT2_DATASET',
        '/tmp/datasets/wikitext-103-gpt2/')
    dataset_url = os.getenv('CI_GPT2_DATASET_URL')

    # If it's not there download it
    if not os.path.exists(dataset_dir):
        # If no URL was provided error out
        if dataset_url is None:
            raise RuntimeError(
                'No dataset found or URL set. '
                'Set CI_GPT2_DATASET_URL environment variable '
                'to an URL where the dataset is available.')

        # Use a lockfile so only one process proceeds
        download_lock_path = Path(build_dir, ".ipu_sparse_ops.gpt2.download.lockfile")
        try:
            with open(download_lock_path, "x") as lockfile:
                pass

            # Create parent directories
            dataset_dir_parent = str(Path(dataset_dir).parent)
            os.makedirs(dataset_dir_parent, exist_ok=True)

            # Scratch space to avoid race conditions
            workspace_dir = tempfile.TemporaryDirectory()
            tar_path = f'{workspace_dir.name}/wikitext-103-gpt2.tar.gz'
            unpacked_dir = f'{workspace_dir.name}/wikitext-103-gpt2'

            # Download dataset
            r = subprocess.run([
                'curl',
                '-o', tar_path,
                f'{dataset_url.rstrip("/")}/wikitext-103-gpt2.tar.gz'
            ])
            if r.returncode != 0:
                raise RuntimeError('Download failed.')

            # Unpack it
            r = subprocess.run(['tar', 'xf', tar_path, '-C', workspace_dir.name])
            if r.returncode != 0:
                raise RuntimeError('Unpack failed.')

            # Move to final destination from scratch space
            shutil.move(unpacked_dir, dataset_dir_parent)
            workspace_dir.cleanup()

        except FileExistsError as e:
            print("\nDataset is already downloading")

    total_seconds = 120
    sleep_seconds = 10
    retries = (total_seconds+sleep_seconds-1)//sleep_seconds
    while not os.path.exists(dataset_dir):
        retries -= 1
        if retries == 0:
            raise RuntimeError("Timed out waiting for gpt2 dataset to download and extract.")
        time.sleep(sleep_seconds)
