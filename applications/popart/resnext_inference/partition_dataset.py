# Copyright 2019 Graphcore Ltd.
import os
import math
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, required=True)
parser.add_argument("--partitions", type=int, default=8)
parser.add_argument("--output", type=str, required=True, default="datasets/")
parser.add_argument("--verbose", action="store_true")
opts = parser.parse_args()

image_filenames = [os.path.join(opts.data_dir, f) for f in
                   os.listdir(opts.data_dir) if not f.startswith('.')]
image_filenames = [f for f in image_filenames if f.lower().endswith(('.jpg', '.jpeg'))]

num_files = len(image_filenames)
files_per_partition = math.floor(num_files/opts.partitions)

os.makedirs(opts.output, exist_ok=True)


def copy(src, dst):
    if src != dst:
        shutil.copy(src, dst)

for i in range(opts.partitions):
    partition_dir = os.path.join(opts.output, str(i), "pytorch_subdir")
    print(f"Partitioned {partition_dir}")
    os.makedirs(partition_dir, exist_ok=True)
    for f in image_filenames[i*files_per_partition:(i+1)*files_per_partition]:
        dst = os.path.join(partition_dir, os.path.basename(f))
        if opts.verbose:
            print(dst)
        copy(f, dst)
