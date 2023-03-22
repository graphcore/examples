# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import argparse
import json
import numpy as np
from tensorflow.python.ipu.dataset_benchmark import dataset_benchmark
from eight_bit_transfer import EightBitTransfer
from datasets.dataset_factory import DatasetFactory
import batch_config

import logging

from configuration.terminal_argparse import str_to_bool
from precision import Precision
import popdist
from tensorflow.python.ipu import distributed
from tensorflow.python.distribute import reduce_util


def add_arguments(parser):
    parser.add_argument("--dataset", type=str, default="cifar10", help="Name of dataset to use")
    parser.add_argument("--dataset-path", type=str, default=".", help='Path to dataset, default="."')
    parser.add_argument(
        "--micro-batch-size", type=int, default=8, help="Micro batch size, in number of samples, default=1"
    )
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs, default=1")
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "test"], help="Which split of a dataset to use"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16.16",
        choices=Precision.supported_precisions,
        help="<compute precision>.<weight update precision> both 16 or 32, default='16.16'",
    )
    parser.add_argument(
        "--eight-bit-transfer",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable/disable input transfer in 8 bit, default=False/Off",
    )
    parser.add_argument(
        "--pipeline-num-parallel",
        type=int,
        default=48,
        help="Number of images to process in parallel on the host side.",
    )
    parser.add_argument(
        "--synthetic-data",
        type=str,
        default=None,
        help="Enable usage of synthetic data on the host or ipu. Corresponding options are 'host' or 'ipu'",
    )
    parser.add_argument(
        "--accelerator-side-preprocess",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="When enabled some preprocessing steps (depending on the chosen dataset), are run "
        "on the accelerator rather on the host.",
    )
    return parser


def estimate_ds_throughput(ds, ds_size: int, epochs: int, micro_batch_size: int, num_instances: int) -> float:
    results_tfdatatype = dataset_benchmark(ds, epochs, elements_per_epochs=ds_size // micro_batch_size // num_instances)
    results_dict = json.loads(results_tfdatatype.numpy()[0].decode("utf-8"))
    throughputs = [epoch["elements_per_second"] * micro_batch_size for epoch in results_dict["epochs"]]
    logging.info(f"All epochs throughputs {throughputs}")
    skip_epochs = min(2, epochs - 1)
    throughputs = throughputs[skip_epochs:]
    mean_throughput = np.mean(throughputs)
    if num_instances > 1:
        mean_throughput = distributed.allreduce(mean_throughput, reduce_util.ReduceOp.SUM)
    return mean_throughput


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TF2 classification dataset benchmark")
    parser = add_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"args = {args}")

    fp_precision = Precision(args.precision)

    eight_bit_transfer = (
        EightBitTransfer(compute_precision=fp_precision.compute_precision) if args.eight_bit_transfer else None
    )

    pipeline_num_parallel = args.pipeline_num_parallel

    # check if the script has been called by poprun
    num_instances = popdist.getNumInstances()

    if num_instances > 1:
        popdist.init()

    if args.synthetic_data == "ipu":
        logging.warn(
            "Synthetic data on ipu not allowed for this benchmark, because datasets are supposed to be on the host. Changing it to host"
        )
        args.synthetic_data = "host"

    batch_config = batch_config.BatchConfig(args.micro_batch_size, num_replicas=1, gradient_accumulation_count=1)

    app_dataset, _, _ = DatasetFactory.get_dataset(
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        split=args.split,
        img_datatype=fp_precision.compute_precision,
        batch_config=batch_config,
        eight_bit_transfer=eight_bit_transfer,
        pipeline_num_parallel=pipeline_num_parallel,
        synthetic_data=args.synthetic_data,
        accelerator_side_preprocess=args.accelerator_side_preprocess,
    )

    throughput = estimate_ds_throughput(
        app_dataset.pipeline, app_dataset.size, args.num_epochs, args.micro_batch_size, num_instances
    )

    logging.info(f"Throughput = {throughput:.2f} examples/sec")
