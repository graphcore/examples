# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import time
import argparse
import copy
from pathlib import Path
import poptorch
import pytest
import numpy as np
import import_helper
import logging
import torch
import popdist.poptorch
import horovod.torch as hvd
import models
import datasets
import utils
import sys


def benchmark(inference_model, test_data, args):
    elapsed_times, sample_sizes = [], []
    min_latencies, max_latencies, avg_latencies = [], [], []

    for input_data, _ in test_data:

        if args.compile_only:
            _ = inference_model.compile(input_data)
            logging.info("Graph compilation complete, --compile-only was set, exiting.")
            sys.exit(0)

        start_time = time.perf_counter()
        result = inference_model(input_data)
        elapsed_time = time.perf_counter() - start_time
        sample_size = input_data.size()[0]
        elapsed_times.append(elapsed_time)
        sample_sizes.append(sample_size)

        min_latency, max_latency, avg_latency = tuple([1000.0 * time for time in inference_model.getLatency()])
        min_latencies.append(min_latency)
        max_latencies.append(max_latency)
        avg_latencies.append(avg_latency)

        if args.use_popdist:
            instance_info = f"[instance{args.popdist_rank + 1}/{args.popdist_size}]"
        else:
            instance_info = ""
        logging.info(f"{instance_info} Throughput: {sample_size / elapsed_time} imgs/sec; Latency range: {min_latency}..{max_latency} ms")

        if args.profile:
            return

    # Remove the first few throughput measurements to eliminate startup overhead.
    elapsed_times = elapsed_times[2:]
    sample_sizes = sample_sizes[2:]

    num_instances = 1
    if args.use_popdist:
        num_instances = args.popdist_size
        synchronized_times = []
        for t in elapsed_times:
            synchronized_times.append(torch.max(hvd.allgather(torch.tensor([t]))).item())
        elapsed_times = synchronized_times
        sample_sizes = [torch.tensor(x) for x in sample_sizes]
        sample_sizes = [t.item() for t in hvd.grouped_allreduce(sample_sizes, op=hvd.Sum)]

    throughputs = [s / t for s, t in zip(sample_sizes, elapsed_times)]

    min_latency = min(min_latencies)
    max_latency = max(max_latencies)
    if args.use_popdist:
        min_latency, max_latency, avg_latencies = utils.synchronize_latency_values(
            min_latency,
            max_latency,
            avg_latencies,
        )
    # 'avg_latency' is average of averages or average of averages of averages when running multiple processes.
    avg_latency = np.mean(avg_latencies)

    logging.info("-------------------------------------------------------------------------------------------")
    # Standardised metric reporting
    logging.info(f"{args.model} results:")
    logging.info(f"\n\tmicro_batch_size: {args.micro_batch_size}")
    logging.info(f"\n\tdata_mode: {args.data}")
    logging.info(f"\n\tdata_type: {result.data.dtype}")
    logging.info(f"\n\tnum_ipus: {num_instances * args.replicas * (len(args.pipeline_splits) + 1)}")
    logging.info(f"\n\tthroughput: {np.mean(throughputs)} samples/sec (mean) (min: {np.min(throughputs)}, max: {np.max(throughputs)}, std: {np.std(throughputs)})")
    logging.info(f"\n\tlatency avg: {avg_latency} ms (mean) (min: {min_latency}, max: {max_latency})")
    if args.data == "synthetic":
        logging.info(f"\n\tNOTE: Using synthetic data, ignore these latency values.")


def parse_arguments():
    common_parser = utils.get_common_parser()
    parser = argparse.ArgumentParser(parents=[common_parser], description='CNN inference in PopTorch')
    parser.add_argument('--data', choices=['real', 'synthetic', 'generated'], default='real', help="Choose data")
    parser.add_argument('--precision', choices=['16.16', '32.32'], default='16.16', help="Precision of Ops(weights/activations/gradients) and Master data types: 16.16, 32.32")
    parser.add_argument('--random-weights', action='store_true', help="When true, weights of the model are initialized randomly")
    args = utils.parse_with_config(parser, Path(__file__).parent.absolute().joinpath("configs.yml"))
    if args.eight_bit_io and args.normalization_location == 'host':
        logging.warning("for eight-bit input, please use IPU-side normalisation, setting normalisation to IPU")
        args.normalization_location = 'ipu'
    utils.handle_distributed_settings(args)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    utils.Logger.setup_logging_folder(args)
    if args.use_popdist:
        opts = popdist.poptorch.Options(
            ipus_per_replica=len(args.pipeline_splits) + 1)
    else:
        opts = poptorch.Options()
        opts.replicationFactor(args.replicas)
    opts.deviceIterations(args.device_iterations)

    dataloader = datasets.get_data(
        args, opts, train=False, async_dataloader=False)
    model = models.get_model(
        args, datasets.datasets_info[args.data], pretrained=not args.random_weights, inference_mode=True)

    opts = utils.inference_settings(args, copy.deepcopy(opts))
    inference_model = poptorch.inferenceModel(model, opts)

    benchmark(inference_model, dataloader, args)
