# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from tqdm import tqdm
import poptorch
import time
import argparse
import popdist
import horovod.torch as hvd
import import_helper
from datasets.dataset import datasets_info, get_data
import utils


def get_args():
    parser = argparse.ArgumentParser(add_help=True, description='Benchmarking the host side throughput')
    parser.add_argument('--batch-size', type=int, default=1024, help='Global batch size')
    parser.add_argument('--data', choices=datasets_info.keys(), default='imagenet', help='Select dataset')
    parser.add_argument('--imagenet-data-path', type=str, default="/localdata/datasets/imagenet-raw-data", help="Path of the raw imagenet data")
    parser.add_argument('--disable-async-loading', action='store_true', help='Not using the async DataLoader')
    parser.add_argument('--normalization-location', choices=['host', 'ipu'], default='host', help='Location of the data normalization')
    parser.add_argument('--dataloader-worker', type=int, default=32, help="Number of worker for each dataloader")
    parser.add_argument('--eight-bit-io', action='store_true', help="Image transfer from host to IPU in 8-bit format, requires normalisation on the IPU")
    parser.add_argument('--dataloader-rebatch-size', type=int, help='Dataloader rebatching size. (Helps to optimise the host memory footprint)')
    parser.add_argument('--webdataset-memory-cache-ratio', type=float, default=0.0, help="Determines the portion of the webdataset, which is cached in memory.")

    args = parser.parse_args()
    args.precision = "16.16"
    args.model = 'resnet50'
    args.seed = 0
    args.use_bbox_info = True
    args.device_iterations = 1
    args.replicas = 1
    args.webdataset_percentage_to_use = 100
    return args


def benchmark_throughput(dataloader, iteration=2):
    for _ in range(iteration):
        total_sample_size = 0
        start_time = time.perf_counter()
        for input_data, _ in tqdm(dataloader, total=len(dataloader)):
            total_sample_size += input_data.size()[0]
        elapsed_time = time.perf_counter() - start_time

        if popdist.isPopdistEnvSet():
            elapsed_time, total_sample_size = utils.synchronize_throughput_values(
                elapsed_time,
                total_sample_size,
            )

        iteration_throughput = total_sample_size / elapsed_time
        print(f"Throughput of the iteration:{iteration_throughput:0.1f} img/sec")


if __name__ == '__main__':
    args = get_args()
    opts = poptorch.Options()
    if popdist.isPopdistEnvSet():
        hvd.init()
        opts.Distributed.configureProcessId(popdist.getInstanceIndex(), popdist.getNumInstances())
    opts.randomSeed(0)
    dataloader = get_data(args, opts, train=True, async_dataloader=not(args.disable_async_loading))
    benchmark_throughput(dataloader)
