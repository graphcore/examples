# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import os
import math
import shutil
import argparse

# Make one data directory per process, and copy the entire dataset into it.

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", help="Location of dataset to copy", type=str, required=True)
parser.add_argument("--copies", help="Number of copies of data to make", type=int, default=4)
parser.add_argument("--output", help="Location of copies of dataset", type=str, required=True, default="datasets/")
parser.add_argument("--verbose", action="store_true")
opts = parser.parse_args()

image_filenames = [os.path.join(opts.data_dir, f) for f in
                   os.listdir(opts.data_dir) if not f.startswith('.')]
image_filenames = [f for f in image_filenames if f.lower().endswith(('.jpg', '.jpeg'))]

num_files = len(image_filenames)

os.makedirs(opts.output, exist_ok=True)


def copy(src, dst):
    if src != dst:
        shutil.copy(src, dst)

for i in range(opts.copies):
    # The torchvision.datasets.ImageFolder class expects the images to be in a directory with nested subdirectory
    partition_dir = os.path.join(opts.output, str(i), "pytorch_subdir")
    print(f"Partitioned {partition_dir}")
    os.makedirs(partition_dir, exist_ok=True)
    for f in image_filenames:
        dst = os.path.join(partition_dir, os.path.basename(f))
        if opts.verbose:
            print(dst)
        copy(f, dst)
