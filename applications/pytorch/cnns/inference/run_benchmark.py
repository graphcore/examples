# Copyright 2020 Graphcore Ltd.
import time
import argparse
import torch
import poptorch
import numpy as np
from data import get_dataloader, datasets_shape
import sys
import logging
sys.path.append('..')
import models


# Set up logging
logging.basicConfig(format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)


def benchmark(inference_model, test_data, opts):
    fps = []
    with torch.no_grad():
        for data in test_data:
            data = data.contiguous()
            start = time.time()
            result = inference_model(data)
            end = time.time()
            fps.append(data.size()[0]/(end-start))
    # Remove the first few and the last to eliminate the compile and the last possibly smaller batch
    fps = fps[2:]
    print("-------------------------------------------------------------------------------------------")
    print("Throughput at bs={}, data_mode={}, data_type={},"
          " num_ipus={}, of {}: min={} imgs/s, max={} imgs/sec, mean={} imgs/sec, std={} imgs/sec.".format(opts.batch_size,
                                                                                                           opts.data,
                                                                                                           result.data.dtype,
                                                                                                           opts.replicas*(len(opts.pipeline_splits)+1),
                                                                                                           opts.model,
                                                                                                           min(fps),
                                                                                                           max(fps),
                                                                                                           np.mean(fps),
                                                                                                           np.std(fps)))


def parse_arguments():
    parser = argparse.ArgumentParser(description='CNN training in PopTorch')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for training (default: 1)')
    parser.add_argument('--model', choices=models.available_models.keys(),  default='resnet18', help="Choose model")
    parser.add_argument('--data', choices=['real', 'synthetic'], default='real', help="Choose data")
    parser.add_argument('--pipeline-splits', type=str, nargs='+', default=[], help="List of the splitting layers")
    parser.add_argument('--replicas', type=int, default=1, help="Number of IPU replicas")
    parser.add_argument('--device-iteration', type=int, default=1, help="Device Iteration")
    parser.add_argument('--precision', choices=['full', 'half'], default='full', help="Floating Point precision")

    opts = parser.parse_args()
    return opts

if __name__ == '__main__':
    opts = parse_arguments()
    logging.info(opts)
    model = models.get_model(opts, datasets_shape[opts.data])
    model.eval()

    data = get_dataloader(opts.batch_size*opts.device_iteration*opts.replicas, opts.data == "synthetic")
    model_opts = poptorch.Options().replicationFactor(opts.replicas) \
                                   .deviceIterations(opts.device_iteration)
    inference_model = poptorch.inferenceModel(model, model_opts)
    benchmark(inference_model, data, opts)
