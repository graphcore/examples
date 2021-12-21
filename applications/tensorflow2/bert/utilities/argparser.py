# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import argparse
import json


def add_arguments(parser):
    parser.add_argument("config",
                        type=str,
                        help="Path to config file")
    parser.add_argument("--dataset-dir",
                        type=str,
                        help="Path to directory containing the dataset")
    parser.add_argument("--save-ckpt-path",
                        type=str,
                        help="Path to directory to save the training checkpoints.")
    parser.add_argument("--pretrained-ckpt-path",
                        type=str,
                        help="Path to a pretrained checkpoint.")
    parser.add_argument("--total-num-train-samples",
                        type=int,
                        help=("Set the number of samples required for full training."))
    parser.add_argument("--name",
                        type=str,
                        help="Optional universal name.")
    return parser


def str_to_bool(value):
    # boolean args can be used as flags to  set value = const
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n', 'off'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y', 'on'}:
        return True
    raise argparse.ArgumentTypeError(f'{value} is not a valid boolean value')


def parse_arguments(description):
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_arguments(parser)
    args = parser.parse_args()

    with open(args.config, "r") as read_file:
        cfg = json.load(read_file)
    # Combine config and arguments, excluding those arguments that
    # haven't been set on the command line.
    return {**cfg, **{k: v for k, v in vars(args).items() if v is not None}}
