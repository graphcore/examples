# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path
from subprocess import run, PIPE
from examples_tests.execute_once_per_fs import ExecuteOncePerFS

import functools
import os
import tempfile
import time
import shutil
import subprocess

import pytest

examples_dir = Path(__file__).parent.parent.parent.parent
build_dir = Path(__file__).parent


@pytest.fixture
def ipu_sparse_ops(scope="session"):
    """This function builds the ipu_sparse_ops
    library for any tests that rely on it.
    """
    build_path = Path(
        examples_dir,
        "sparsity",
        "dynamic_sparsity",
        "tensorflow1",
    )
    completed = run(['python3-config', '--extension-suffix'], stdout=PIPE)
    extension = completed.stdout.decode().replace('\n', '')
    shared_libs = [f'host_utils{extension}', 'libsparse_matmul.so']
    paths = [Path(build_path, "ipu_sparse_ops", f) for f in shared_libs]

    # Use exclusive lockfile to avoid race conditions on the build:
    lock_path = Path(build_path, ".ipu_sparse_ops.pytest.build.lockfile")

    @ExecuteOncePerFS(lockfile=lock_path, file_list=paths, timeout=120, retries=20)
    def build_dynsparse():
        run(['make', 'clean'], cwd=build_path)
        run(['make', '-j'], cwd=build_path)

    build_dynsparse()


@pytest.fixture
def wikitext_103_dataset(scope="session"):
    """Checks that wikitext-103 is available and if not downloads it."""
    # Lookup dataset storage location
    dataset_dir = os.getenv(
        'CI_GPT2_DATASET',
        '/tmp/datasets/wikitext-103-gpt2/')
    dataset_url = os.getenv('CI_GPT2_DATASET_URL')

    # If it's there we return immediately
    if os.path.exists(dataset_dir):
        return

    # If it's not there attempt to download it
    # Error out if no URL was provided
    if dataset_url is None:
        raise RuntimeError(
            'No dataset found or URL set. '
            'Set CI_GPT2_DATASET_URL environment variable '
            'to an URL where the dataset is available.')

    # Use a lockfile so only one process proceeds
    download_lock_path = Path(build_dir, ".ipu_sparse_ops.gpt2.download.lockfile")

    @ExecuteOncePerFS(lockfile=download_lock_path, file_list=[dataset_dir], timeout=120, retries=20)
    def download_and_extract():
        # Create parent directories
        dataset_dir_parent = str(Path(dataset_dir).parent)
        os.makedirs(dataset_dir_parent, exist_ok=True)

        # Scratch space to avoid race conditions
        with tempfile.TemporaryDirectory() as workspace_dir:
            tar_path = f'{workspace_dir}/wikitext-103-gpt2.tar.gz'
            unpacked_dir = f'{workspace_dir}/wikitext-103-gpt2'

            # Download dataset
            r = subprocess.run([
                'curl',
                '-o', tar_path,
                f'{dataset_url.rstrip("/")}/wikitext-103-gpt2.tar.gz'
            ])
            if r.returncode != 0:
                raise RuntimeError('Download failed.')

            # Unpack it
            r = subprocess.run(['tar', 'xf', tar_path, '-C', workspace_dir])
            if r.returncode != 0:
                raise RuntimeError('Unpack failed.')

            # Move to final destination from scratch space
            shutil.move(unpacked_dir, dataset_dir_parent)

    download_and_extract()
