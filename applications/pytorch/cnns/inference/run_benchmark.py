# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import time
import argparse
from pathlib import Path
import torch
import poptorch
import numpy as np
import sys
import logging
import popdist.poptorch
sys.path.append('..')
import models
import datasets
import utils


def benchmark(inference_model, test_data, opts):
    fps = []
    latencies = {'min': [],
                 'max': [],
                 'avg': []}
    with torch.no_grad():
        for input_data, label in test_data:
            start = time.time()
            result = inference_model(input_data)
            end = time.time()
            tput = input_data.size()[0]/(end-start)
            fps.append(tput)
            if not opts.data == "synthetic":   # There is no latency for synthetic data.
                min_latency, max_latency, avg_latency = tuple([1000.0 * time for time in inference_model.getLatency()])
                latencies['min'].append(min_latency)
                latencies['max'].append(max_latency)
                latencies['avg'].append(avg_latency)
            else:
                min_latency, max_latency, avg_latency = "N/A", "N/A", "N/A"
            if opts.use_popdist:
                instance_info = f"[instance{opts.popdist_rank + 1}/{opts.popdist_size}]"
            else:
                instance_info = ""
            logging.info(f"{instance_info} Throughput: {tput} imgs/sec; Latency range: {min_latency}..{max_latency} ms")

    # Remove the first few measurements to eliminate startup overhead
    fps = fps[2:]
    # Calculate latency statistics
    if opts.data == "synthetic":
        min_latency, max_latency, avg_latency = "N/A", "N/A", "N/A"
    else:
        min_latency = min(latencies['min'])
        max_latency = max(latencies['max'])
        avg_latency = np.mean(latencies['avg'])
    # Sync the values if distributed
    if opts.use_popdist:
        fps = utils.sync_metrics(fps, average=False)
        if not opts.data == "synthetic":
            min_latency = utils.sync_metrics(min_latency, average=True)
            max_latency = utils.sync_metrics(max_latency, average=True)
            avg_latency = utils.sync_metrics(avg_latency, average=True)
    logging.info("-------------------------------------------------------------------------------------------")
    logging.info(f"Throughput at bs={opts.batch_size}, data_mode={opts.data}, data_type={result.data.dtype},"
                 f" num_ipus={opts.replicas*(len(opts.pipeline_splits)+1)}, of {opts.model}: min={min(fps)} imgs/s, max={max(fps)} imgs/s, mean={np.mean(fps)} imgs/s, std={np.std(fps)} imgs/s.")
    logging.info(f"Latency: min={min_latency} ms, max={max_latency} ms, mean={avg_latency} ms.")


def parse_arguments():
    common_parser = utils.get_common_parser()
    parser = argparse.ArgumentParser(parents=[common_parser], description='CNN inference in PopTorch')
    parser.add_argument('--data', choices=['real', 'synthetic', 'generated'], default='real', help="Choose data")
    parser.add_argument('--precision', choices=['16.16', '32.32'], default='16.16', help="Precision of Ops(weights/activations/gradients) and Master data types: 16.16, 32.32")
    opts = utils.parse_with_config(parser, Path(__file__).parent.absolute().joinpath("configs.yml"))
    if opts.eight_bit_io and opts.normalization_location == 'host':
        logging.warning("for eight-bit input, please use IPU-side normalisation, setting normalisation to IPU")
        opts.normalization_location = 'ipu'
    utils.handle_distributed_settings(opts)
    return opts


if __name__ == '__main__':
    opts = parse_arguments()
    utils.Logger.setup_logging_folder(opts)
    if opts.use_popdist:
        model_opts = popdist.poptorch.Options(ipus_per_replica=len(opts.pipeline_splits) + 1)
    else:
        model_opts = poptorch.Options()
        model_opts.replicationFactor(opts.replicas)
    model_opts.deviceIterations(opts.device_iterations)

    dataloader = datasets.get_data(opts, model_opts, train=False, async_dataloader=False)
    model = models.get_model(opts, datasets.datasets_info[opts.data], pretrained=True)
    model.eval()

    model_opts = utils.inference_settings(opts, model_opts)
    inference_model = poptorch.inferenceModel(model, model_opts)

    benchmark(inference_model, dataloader, opts)
