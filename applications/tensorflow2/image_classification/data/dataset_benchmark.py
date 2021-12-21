# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import argparse
import json
import numpy as np
from tensorflow.python.ipu.dataset_benchmark import dataset_benchmark
from eight_bit_transfer import EightBitTransfer
from data.dataset_factory import DatasetFactory

import logging

from configuration.terminal_argparse import str_to_bool
from precision import Precision


def add_arguments(parser):
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Name of dataset to use')
    parser.add_argument('--dataset-path', type=str, default='.',
                        help='Path to dataset, default="."')
    parser.add_argument('--micro-batch-size', type=int, default=8,
                        help="Micro batch size, in number of samples, default=1")
    parser.add_argument('--num-epochs', type=int, default=1,
                        help="Number of training epochs, default=1")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help="Which split of a dataset to use")
    parser.add_argument('--precision', type=str, default='16.16', choices=Precision.supported_precisions,
                        help="<compute precision>.<weight update precision> both 16 or 32, default='16.16'")
    parser.add_argument('--eight-bit-transfer', type=str_to_bool, nargs="?", const=True, default=False,
                        help='Enable/disable input transfer in 8 bit, default=False/Off')
    return parser


def estimate_ds_throughput(ds, ds_size: int, epochs: int, micro_batch_size: int) -> float:
    results_tfdatatype = dataset_benchmark(ds, epochs, elements_per_epochs=ds_size / micro_batch_size)
    results_dict = json.loads(results_tfdatatype.numpy()[0].decode('utf-8'))
    mean_throughput = np.mean([epoch['elements_per_second'] for epoch in results_dict['epochs']]) * micro_batch_size
    return mean_throughput


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TF2 classification dataset benchmark')
    parser = add_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f'args = {args}')

    fp_precision = Precision(args.precision)

    eight_bit_transfer = EightBitTransfer(
        compute_precision=fp_precision.compute_precision) if args.eight_bit_transfer else None

    ds, _, ds_size, _ = DatasetFactory.get_dataset(
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        split=args.split,
        img_datatype=fp_precision.compute_precision,
        micro_batch_size=args.micro_batch_size,
        eight_bit_transfer=eight_bit_transfer
    )

    throughput = estimate_ds_throughput(ds, ds_size, args.num_epochs, args.micro_batch_size)

    logging.info(f'Throughput = {throughput:.2f} examples/sec')
