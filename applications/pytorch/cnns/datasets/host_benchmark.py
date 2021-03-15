# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from tqdm import tqdm
import poptorch
import time
import argparse
import sys
sys.path.append('..')
from datasets.dataset import datasets_info, get_data


def get_args():
    parser = argparse.ArgumentParser(add_help=True, description='Benchmarking the host side throughput')
    parser.add_argument('--batch-size', type=int, default=1024, help='Global batch size')
    parser.add_argument('--data', choices=datasets_info.keys(), default='imagenet', help='Select dataset')
    parser.add_argument('--imagenet-data-path', type=str, default="/localdata/datasets/imagenet-raw-data", help="Path of the raw imagenet data")
    parser.add_argument('--disable-async-loading', action='store_true', help='Not using the async DataLoader')
    parser.add_argument('--normalization-location', choices=['host', 'ipu'], default='host', help='Location of the data normalization')
    parser.add_argument('--dataloader-worker', type=int, default=32, help="Number of worker for each dataloader")
    args = parser.parse_args()
    args.precision = "16.16"
    args.model = 'resnet50'
    args.device_iterations = 1
    args.replicas = 1
    return args


def benchmark_throughput(dataloader, iteration=2):
    for iter in range(iteration):
        total_sample = 0
        start_time = time.time()
        iterations_per_epoch = len(dataloader)
        bar = tqdm(dataloader, total=iterations_per_epoch)
        for input_data, label in bar:
            total_sample += input_data.size()[0]
        end_time = time.time()
        epoch_throughput = total_sample / (end_time-start_time)
        print(f"Throughput of the epoch:{epoch_throughput:0.1f} img/sec")


if __name__ == '__main__':
    opts = get_args()
    model_opts = poptorch.Options()
    model_opts.randomSeed(0)
    dataloader = get_data(opts, model_opts, train=True, async_dataloader=not(opts.disable_async_loading))
    benchmark_throughput(dataloader)
