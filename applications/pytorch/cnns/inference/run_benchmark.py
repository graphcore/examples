# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import time
import argparse
import torch
import poptorch
import numpy as np
import sys
import logging
sys.path.append('..')
import models
import data
import utils


def benchmark(inference_model, test_data, opts):
    fps = []
    with torch.no_grad():
        for input_data, label in test_data:
            start = time.time()
            result = inference_model(input_data)
            end = time.time()
            fps.append(input_data.size()[0]/(end-start))
    # Remove the first few measurements to eliminate startup overhead
    fps = fps[2:]
    logging.info("-------------------------------------------------------------------------------------------")
    logging.info(f"Throughput at bs={opts.batch_size}, data_mode={opts.data}, data_type={result.data.dtype},"
                 f" num_ipus={opts.replicas*(len(opts.pipeline_splits)+1)}, of {opts.model}: min={min(fps)} imgs/s, max={max(fps)} imgs/s, mean={np.mean(fps)} imgs/s, std={np.std(fps)} imgs/s.")


def parse_arguments():
    common_parser = utils.get_common_parser()
    parser = argparse.ArgumentParser(parents=[common_parser], description='CNN inference in PopTorch')
    parser.add_argument('--data', choices=['real', 'synthetic'], default='real', help="Choose data")
    parser.add_argument('--iterations', type=int, default=100, help='number of program iterations')
    parser.add_argument('--precision', choices=['16.16', '32.32'], default='16.16', help="Precision of Ops(weights/activations/gradients) and Master data types: 16.16, 32.32")
    opts = utils.parse_with_config(parser, "configs.yml")
    return opts


if __name__ == '__main__':
    opts = parse_arguments()
    utils.Logger.setup_logging_folder(opts)
    model_opts = poptorch.Options()
    model_opts.replicationFactor(opts.replicas)
    model_opts.deviceIterations(opts.device_iterations)

    dataloader = data.get_data(opts, model_opts, train=False, async_dataloader=False)
    model = models.get_model(opts, data.datasets_info[opts.data], pretrained=True)
    model.eval()

    if opts.data == "synthetic":
        model_opts.Popart.set("syntheticDataMode", 2)
    if opts.half_partial:
        model_opts.Popart.set("partialsTypeMatMuls", "half")
        model_opts.Popart.set("convolutionOptions", {'partialsType': 'half'})

    inference_model = poptorch.inferenceModel(model, model_opts)
    benchmark(inference_model, dataloader, opts)
