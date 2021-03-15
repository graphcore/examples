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
import datasets
import utils


def benchmark(inference_model, test_data, opts):
    fps = []
    step_time = []
    with torch.no_grad():
        for input_data, label in test_data:
            start = time.time()
            result = inference_model(input_data)
            end = time.time()
            tput = input_data.size()[0]/(end-start)
            elapsed_time = (end-start) * 1000.0
            fps.append(tput)
            step_time.append(elapsed_time)
            logging.info(f"Throughput: {tput} imgs/sec; Step time: {elapsed_time} ms")

    # Remove the first few measurements to eliminate startup overhead
    fps = fps[2:]
    step_time = step_time[2:]
    logging.info("-------------------------------------------------------------------------------------------")
    logging.info(f"Throughput at bs={opts.batch_size}, data_mode={opts.data}, data_type={result.data.dtype},"
                 f" num_ipus={opts.replicas*(len(opts.pipeline_splits)+1)}, of {opts.model}: min={min(fps)} imgs/s, max={max(fps)} imgs/s, mean={np.mean(fps)} imgs/s, std={np.std(fps)} imgs/s.")
    logging.info(f"Step time: min={min(step_time)} ms, max={max(step_time)} ms, mean={np.mean(step_time)} ms, std={np.std(step_time)} ms.")


def parse_arguments():
    common_parser = utils.get_common_parser()
    parser = argparse.ArgumentParser(parents=[common_parser], description='CNN inference in PopTorch')
    parser.add_argument('--data', choices=['real', 'synthetic', 'generated'], default='real', help="Choose data")
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

    dataloader = datasets.get_data(opts, model_opts, train=False, async_dataloader=False)
    model = models.get_model(opts, datasets.datasets_info[opts.data], pretrained=True)
    model.eval()

    model_opts = utils.inference_settings(opts, model_opts)
    inference_model = poptorch.inferenceModel(model, model_opts)

    benchmark(inference_model, dataloader, opts)
