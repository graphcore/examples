# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import json
import os
import argparse
import webdataset as wds
import torch
import import_helper
from datasets import decode_webdataset, encode_sample


def identity(x):
    return x


def get_args():
    parser = argparse.ArgumentParser(add_help=True, description='Extend WebDataset format for distributed')
    parser.add_argument('--target', type=str, required=True, help='Path of the WebDataset')
    parser.add_argument("--num-instances", required=True, type=int, help='Number of instances')
    args = parser.parse_args()
    return args


def distribute_remaining_data(data_path, subset, total_instance, chunks):
    remaining_chunks = len(chunks) % total_instance
    if remaining_chunks == 0:
        remaining_chunks = total_instance
    remaining_start, remaining_end = chunks[-remaining_chunks], chunks[-1]
    chunk_str = f"{remaining_start}..{remaining_end}"
    dataset = wds.Dataset(os.path.join(data_path, subset + "-{" + chunk_str + "}.tar"))
    # determine saving format
    with open(os.path.join(data_path, "metadata.json")) as metadata_file:
        metadata = json.load(metadata_file)
        data_format = metadata["format"]
    dataset = decode_webdataset(dataset, data_format, identity, use_bbox_info=True)

    folder = os.path.join(data_path, "distributed", str(total_instance) + "-instances")
    if not os.path.exists(folder):
        os.makedirs(folder)
    distributed_save_path = os.path.join(folder, subset + "-%06d.tar")
    save_remaining_data(dataset, distributed_save_path, total_instance)


def save_remaining_data(dataset, save_pattern, total_instance):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=4)
    # count batches
    count = 0
    for data in dataloader:
        count += 1
    # Write shards
    tar_writer(total_instance, dataloader, count, save_pattern)


def create_distributed_remaining(opts, subset):
    chunks = [file_name[-10:-4] for file_name in os.listdir(opts.target) if file_name.startswith(subset)]
    chunks.sort()
    distribute_remaining_data(opts.target, subset, opts.num_instances, chunks)


def tar_writer(instance, dataloader, dataset_size, save_pattern):
    dl = iter(dataloader)
    for instance_id in range(instance):
        sample_count = dataset_size // instance
        if dataset_size % instance > instance_id:
            sample_count += 1
        save_path = save_pattern % instance_id
        with wds.TarWriter(save_path) as sink:
            for sample_id in range(sample_count):
                data, label = next(dl)
                if isinstance(data, tuple) or isinstance(data, list):
                    data, bbox = data
                else:
                    bbox = None
                sink.write(encode_sample(data, label, dataset_size*instance_id+sample_id, bbox=bbox))


if __name__ == '__main__':
    opts = get_args()
    create_distributed_remaining(opts, "train")
    create_distributed_remaining(opts, "validation")
