# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import logging
import requests
from pathlib import Path
import zipfile

from tqdm import tqdm


logger = logging.getLogger("dataset-encode")


DATASET_URLS = {
    "wikitext-2": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip",
    "wikitext-103": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
}

DATASET_FILENAMES = {
    "wikitext-2": "wikitext-2-raw-v1.zip",
    "wikitext-103": "wikitext-103-raw-v1.zip",
}


def get_zip_path(dataset_name, dataset_dir):
    return Path(dataset_dir) / DATASET_FILENAMES[dataset_name]


def get_dataset_path(dataset_name, dataset_dir):
    return Path(dataset_dir) / f"{dataset_name}-raw"


def download_raw_dataset(dataset_name, dataset_dir):
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    zip_path = Path(dataset_dir) / DATASET_FILENAMES[dataset_name]

    response = requests.get(DATASET_URLS[dataset_name], stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    logger.info(f"Downloading {dataset_name}...")
    with open(zip_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError(
            f"File didn't download correctly (expected {total_size} bytes and got {progress_bar.n} bytes).")


def extract_dataset(dataset_name, dataset_dir):
    logger.info(f"Extracting {dataset_name}...")
    with zipfile.ZipFile(get_zip_path(dataset_name, dataset_dir), 'r') as zf:
        # Loop over each file
        for file in tqdm(iterable=zf.namelist(), total=len(zf.namelist())):
            zf.extract(member=file, path=dataset_dir)


def download_and_extract(dataset_name, dataset_dir, force=False):
    if get_dataset_path(dataset_name, dataset_dir).exists() and not force:
        logger.info("Dataset already downloaded, will not overwrite")
        return

    zip_path = get_zip_path(dataset_name, dataset_dir)

    if not zip_path.exists():
        download_raw_dataset(dataset_name, dataset_dir)


    extract_dataset(dataset_name, dataset_dir)
