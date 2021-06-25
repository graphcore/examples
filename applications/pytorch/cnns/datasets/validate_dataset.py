# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import argparse
import json
from checksumdir import dirhash


imagenet_raw_train_checksum = "17aed9504492852ddcc2ac247413aa95"
imagenet_raw_validation_checksum = "8d6a57a773cf3c3354c92155a5b1d5bd"


def validate(path):
    if os.path.exists(os.path.join(path, 'metadata.json')):
        with open(os.path.join(path, "metadata.json")) as metadata_file:
            metadata = json.load(metadata_file)
            expected_checksum = metadata["checksum"]
        checksum = dirhash(path, excluded_files=["metadata.json"])
        if checksum == expected_checksum:
            print("Dataset OK.")
        else:
            print("Dataset is corrupted.")
    else:
        train_checksum = dirhash(os.path.join(path, "train"))
        if train_checksum == imagenet_raw_train_checksum:
            print("Train data OK.")
        else:
            print("Train data is corrupted.")

        validation_checksum = dirhash(os.path.join(path, "validation"))
        if validation_checksum == imagenet_raw_validation_checksum:
            print("Validation data OK.")
        else:
            print("Validation data is corrupted.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True, description='Validate ImageNet Dataset')
    parser.add_argument('--imagenet-data-path', type=str, required=True, help='Imagenet path')
    args = parser.parse_args()
    validate(args.imagenet_data_path)
