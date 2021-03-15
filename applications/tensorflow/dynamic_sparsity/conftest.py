# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path
from subprocess import run, PIPE

import functools
import os
import tempfile
import time
import shutil
import subprocess

import pytest

public_examples_dir = Path(__file__).parent.parent.parent.parent
build_dir = Path(__file__).parent


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
