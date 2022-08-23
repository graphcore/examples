# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse
import copy
import logging
import numpy as np
from pathlib import Path
import poptorch
import pytest
import utils


def get_model_settings(args_to_parse):
    common_parser = utils.get_common_parser()
    parser = argparse.ArgumentParser(
        parents=[common_parser], description='CNN inference in PopTorch')
    parser.add_argument(
        '--data', choices=['real', 'synthetic', 'generated'], default='real', help="Choose data")
    parser.add_argument('--precision', choices=['16.16', '32.32'], default='16.16',
                        help="Precision of Ops(weights/activations/gradients) and Master data types: 16.16, 32.32")
    parser.add_argument('--random-weights', action='store_true',
                        help="When true, weights of the model are initialized randomly")
    args = utils.parse_with_config(parser, Path(
        __file__).parent.parent.parent.absolute().joinpath("inference/configs.yml"), custom_args=args_to_parse)
    if args.eight_bit_io and args.normalization_location == 'host':
        logging.warning(
            "for eight-bit input, please use IPU-side normalisation, setting normalisation to IPU")
        args.normalization_location = 'ipu'
    utils.handle_distributed_settings(args)

    utils.Logger.setup_logging_folder(args)
    if args.use_popdist:
        pytest.fail("use_popdist is unsupported!")
    else:
        opts = poptorch.Options()
        opts.replicationFactor(args.replicas)
    opts.deviceIterations(args.device_iterations)

    opts = utils.inference_settings(args, copy.deepcopy(opts))

    return args, opts


def log_performance_results(args, model_name, request_type, data_type, number_of_processes, throughputs, latencies):
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    num_instances = 1

    logging.info(
        "-------------------------------------------------------------------------------------------")
    # Standardised metric reporting
    logging.info(f"{model_name}-{request_type}-{number_of_processes} results:")
    logging.info(f"\n\tbatch_size: {args.batch_size}")
    logging.info(f"\n\tdata_mode: {args.data}")
    logging.info(f"\n\tdata_type: {data_type}")
    logging.info(
        f"\n\tnum_ipus: {num_instances * args.replicas * (len(args.pipeline_splits) + 1)}")
    logging.info(
        f"\n\tthroughput: {np.mean(throughputs)} samples/sec (mean) (min: {np.min(throughputs)},"
        f" max: {np.max(throughputs)}, std: {np.std(throughputs)})")
    logging.info(
        f"\n\tlatency: {avg_latency} ms (mean) (min: {min_latency}, max: {max_latency})")
