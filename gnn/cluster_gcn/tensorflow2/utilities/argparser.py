# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse
import json

from utilities.options import ALLOWED_DATASET_TYPE, ALLOWED_DEVICE_OPTIONS, ALLOWED_LOGGING_TYPE, ALLOWED_PRECISION_TYPE


def add_arguments(parser):
    parser.add_argument("config",
                        type=str,
                        help="Path to config file")
    parser.add_argument("--training.clusters-per-batch",
                        type=int,
                        help="Number of clusters per batch into for training.")
    parser.add_argument("--training.num-clusters",
                        type=int,
                        help="Number of total clusters to cluster the training dataset into.")
    parser.add_argument("--training.replicas",
                        type=int,
                        help="Number of replicas during training.")
    parser.add_argument("--validation.clusters-per-batch",
                        type=int,
                        help="Number of clusters per batch into for validation.")
    parser.add_argument("--validation.num-clusters",
                        type=int,
                        help="Number of total clusters to cluster the validation dataset into.")
    parser.add_argument("--model.hidden-size",
                        type=int,
                        help="The hidden size for each layer.")
    parser.add_argument("--model.num-layers",
                        type=int,
                        help="Number of hidden layers.")
    parser.add_argument("--model.dropout",
                        type=float,
                        help="Dropout probability (a float between 0.0, and 1.0).")
    parser.add_argument("--training.epochs",
                        type=int,
                        help="Number of epochs to train for.")
    parser.add_argument("--training.executions-per-epoch",
                        type=int,
                        help="Number of executions in one epoch during training.")
    parser.add_argument("--training.epochs-per-execution",
                        type=int,
                        help="Number of epochs per execution during training. Can be used to scale"
                             " up the number of trained samples in one epoch and reduce the number"
                             " of communication between host and device.")
    parser.add_argument("--training.lr",
                        type=float,
                        help="Float value for the learning rate.")
    parser.add_argument("--training.use-sparse-representation",
                        type=str_to_bool,
                        help="Enable sparse mode.")
    parser.add_argument("--training.precision",
                        type=str,
                        choices=ALLOWED_PRECISION_TYPE,
                        help="Specify the precision type")
    parser.add_argument("--training.device",
                        type=str,
                        choices=ALLOWED_DEVICE_OPTIONS,
                        help="Specify the training device type")
    parser.add_argument("--do-training",
                        type=str_to_bool,
                        help="Enables/disables the training.")
    parser.add_argument("--do-validation",
                        type=str_to_bool,
                        help="Enables/disables the validation.")
    parser.add_argument("--do-test",
                        type=str_to_bool,
                        help="Enables/disables the test.")
    parser.add_argument("--wandb",
                        type=str_to_bool,
                        help="Enables logging to Weight & Biases.")
    parser.add_argument("--seed",
                        type=int,
                        help="Set a seed for the model.")
    parser.add_argument("--data-path",
                        type=str,
                        help="Path for the dataset.")
    parser.add_argument("--dataset-name",
                        type=str,
                        choices=ALLOWED_DATASET_TYPE,
                        help="Select dataset to use.")
    parser.add_argument("--regenerate-dataset-cache",
                        type=str_to_bool,
                        help=("If enabled, the preprocessed dataset cache will"
                              " be regenerated."))
    parser.add_argument("--save-dataset-cache",
                        type=str_to_bool,
                        help=("If enabled, the preprocessed dataset will"
                              " be saved."))
    parser.add_argument("--regenerate-clustering-cache",
                        type=str_to_bool,
                        help=("If enabled, the cluster cache will be regenerated."))
    parser.add_argument("--save-clustering-cache",
                        type=str_to_bool,
                        help="If enabled, the clustering will be saved.")
    parser.add_argument("--logging",
                        type=str,
                        choices=ALLOWED_LOGGING_TYPE,
                        help="Specify the logging level")
    parser.add_argument("--save-ckpt-path",
                        type=str,
                        help="Path to directory to save the training checkpoints.")
    parser.add_argument("--load-ckpt-path",
                        type=str,
                        help="Path to a load a checkpoint from.")
    parser.add_argument("--compile-only",
                        action="store_true",
                        help="Enable compile only mode.")
    parser.add_argument("--calculate-cluster-statistics",
                        type=str_to_bool,
                        help=("Enable/disable statistics of the clusters"
                              " being calculated at run time. For large"
                              " graphs this can take a while."))
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


def merge_args_with_options(args_dict, config):
    """
    Merges a dictionary of arguments with a config.
    :param args_dict: A dictionary where the keys represent
        the argument key and value the argument value. Can
        be in the form 'key_1.key_2.key_3': 2, where the
        key is nested value represented by '.'s between the keys.
        This cannot be a nested dictionary, but instead nesting
        should be represented as described above.
    :param config: A dictionary of key and values in the config,
        can be a nested dictionary.
    :return config: A dictionary of the merged args_dict and
        config.
    """
    for args_key, args_val in args_dict.items():
        if args_val is None:
            continue
        args_key_split_list = args_key.split(".")
        temp_dict = config
        for i, args_key_split in enumerate(args_key_split_list):
            if i == len(args_key_split_list) - 1:
                temp_dict[args_key_split] = args_val
                break
            if args_key_split not in temp_dict:
                temp_dict[args_key_split] = dict()
            temp_dict = temp_dict[args_key_split]
    return config


def combine_config_file_with_args(args, option_object):
    with open(args.config, "r") as read_file:
        config = json.load(read_file)
    merged_options = merge_args_with_options(vars(args), config)
    return option_object(**merged_options)
