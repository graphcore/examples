# Copyright 2020 Graphcore Ltd.
import time
import argparse
import torch
import poptorch
import numpy as np
from data import get_dataloader, datasets_info
import sys
import logging
sys.path.append('..')
import models  # noqa: E402
import utils  # noqa: E402


# Set up logging
logging.basicConfig(format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)


def benchmark(inference_model, test_data, opts):
    fps = []
    with torch.no_grad():
        for data in test_data:
            # Convert the input for the correct precision
            if opts.precision == "half":
                data = data.half()
            data = data.contiguous()
            start = time.time()
            result = inference_model(data)
            end = time.time()
            fps.append(data.size()[0]/(end-start))
    # Remove the first few measurements to eliminate startup overhead
    fps = fps[2:]
    print("-------------------------------------------------------------------------------------------")
    print("Throughput at bs={}, data_mode={}, data_type={},"
          " num_ipus={}, of {}: min={} imgs/s, max={} imgs/s, mean={} imgs/s, std={} imgs/s.".format(opts.batch_size,
                                                                                                     opts.data,
                                                                                                     result.data.dtype,
                                                                                                     opts.replicas*(len(opts.pipeline_splits)+1),
                                                                                                     opts.model,
                                                                                                     min(fps),
                                                                                                     max(fps),
                                                                                                     np.mean(fps),
                                                                                                     np.std(fps)))


def parse_arguments():
    common_parser = utils.get_common_parser()
    parser = argparse.ArgumentParser(parents=[common_parser], description='CNN inference in PopTorch')
    parser.add_argument('--data', choices=['real', 'synthetic'], default='real', help="Choose data")
    parser.add_argument('--iterations', type=int, default=100, help='number of program iterations')
    opts = parser.parse_args()
    return opts


if __name__ == '__main__':
    opts = parse_arguments()
    logging.info(opts)
    utils.setup_logging_folder(opts)
    model = models.get_model(opts, datasets_info[opts.data], pretrained=True)
    model.eval()

    batch_size = opts.batch_size*opts.replicas*opts.device_iteration
    model_opts = poptorch.Options()
    model_opts.replicationFactor(opts.replicas)
    model_opts.deviceIterations(opts.device_iteration)
    data = get_dataloader(opts.batch_size, model_opts, opts.iterations, opts.data == "synthetic")

    if opts.data == "synthetic":
        model_opts.Popart.set("syntheticDataMode", 2)
    if opts.half_partial:
        model_opts.Popart.set("partialsTypeMatMuls", "half")
        model_opts.Popart.set("convolutionOptions", {'partialsType': 'half'})

    num_stages = len(opts.pipeline_splits)+1
    if len(opts.available_memory_proportion) == 1:
        model_opts.setAvailableMemoryProportion({f'IPU{i}': opts.available_memory_proportion[0] for i in range(num_stages)})
    elif len(opts.available_memory_proportion) > 1:
            model_opts.setAvailableMemoryProportion({f'IPU{i}': amp for i, amp in enumerate(opts.available_memory_proportion)})

    inference_model = poptorch.inferenceModel(model, model_opts)
    benchmark(inference_model, data, opts)
