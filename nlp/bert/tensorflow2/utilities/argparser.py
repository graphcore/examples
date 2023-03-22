# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import argparse
import json


ALLOWED_GLUE_TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]


def add_shared_arguments(parser):
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--dataset-dir", type=str, help="Path to directory containing the dataset")
    parser.add_argument("--save-ckpt-path", type=str, help="Path to directory to save the training checkpoints.")
    parser.add_argument(
        "--pretrained-ckpt-path",
        type=str,
        help=(
            "Path to a pretrained checkpoint or dir "
            "containing checkpoints. If a path to a specific "
            "checkpoint is passed, that checkpoint will be "
            "loaded, otherwise if a dir containing multiple "
            "checkpoints is passed, the latest one by "
            "timestep will be loaded."
        ),
    )
    parser.add_argument(
        "--total-num-train-samples", type=int, help="Set the number of samples required for full training."
    )
    parser.add_argument(
        "--global-batch.replicas", type=int, help="Set the number of replicas of the graph that will be used."
    )
    parser.add_argument(
        "--global-batch.micro-batch-size", type=int, help="Set the batch size per replica per gradient calculation."
    )
    parser.add_argument(
        "--global-batch.grad-acc-steps-per-replica",
        type=int,
        help="Set the number of gradient calculations accumulated per weight update.",
    )
    parser.add_argument(
        "--ipu-config.matmul-available-memory-proportion-per-pipeline-stage",
        type=list,
        help="Set the memory proportion per pipeline stage available for matmuls.",
    )
    parser.add_argument("--name", type=str, help="Optional universal name.")
    parser.add_argument(
        "--logging",
        type=str,
        choices=["DEBUG", "INFO", "ERROR", "CRITICAL", "WARNING"],
        help="Specify the logging level",
    )
    parser.add_argument("--enable-wandb", type=str_to_bool, help="Enable Weights and Biases logging.")
    parser.add_argument("--wandb-entity-name", type=str, help="Weights and Biases entity name")
    parser.add_argument("--wandb-project-name", type=str, help="Weights and Biases project name")
    parser.add_argument(
        "--generated-dataset", type=str_to_bool, help="Enable generated dataset for pretraining and fine-tuning."
    )
    parser.add_argument("--compile-only", action="store_true", help="Enable compile only mode.")
    return parser


def add_squad_arguments(parser):
    parser.add_argument("--do-training", type=str_to_bool, help="Enables/disables the training phase of fine tuning")
    parser.add_argument(
        "--do-validation", type=str_to_bool, help="Enables/disables the validation phase of fine tuning"
    )
    return parser


def add_glue_arguments(parser):
    parser.add_argument(
        "--glue-task", type=str, choices=ALLOWED_GLUE_TASKS, help="Tasks supported for GLUE fine-tuning."
    )
    parser.add_argument("--do-training", type=str_to_bool, help="Enables/disables the training phase of fine tuning")
    parser.add_argument(
        "--do-validation", type=str_to_bool, help="Enables/disables the validation phase of fine tuning"
    )
    parser.add_argument(
        "--do-test",
        type=str_to_bool,
        help="Enables/disables the validation phase of fine tuning on the hold out test set",
    )
    return parser


def str_to_bool(value):
    # boolean args can be used as flags to  set value = const
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {"false", "f", "0", "no", "n", "off"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y", "on"}:
        return True
    raise argparse.ArgumentTypeError(f"{value} is not a valid boolean value")


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
        cfg = json.load(read_file)
    merged_options = merge_args_with_options(vars(args), cfg)
    # Combine config and arguments, excluding those arguments that
    # haven't been set on the command line.
    return option_object(**{**merged_options, **{k: v for k, v in vars(args).items() if v is not None}})
