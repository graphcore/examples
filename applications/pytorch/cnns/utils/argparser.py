# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import sys
import yaml
sys.path.append('..')
import models


class YAMLNamespace(argparse.Namespace):
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def get_available_configs(config_file):
    with open(config_file) as file:
        configs = yaml.full_load(file)
    return configs


def parse_with_config(parser, config_file):
    configurations = get_available_configs(config_file)
    parser.add_argument('--config', choices=configurations.keys(), help="Select from avalible configurations")
    args = parser.parse_args()
    if args.config is not None:
        # Load the configurations from the YAML file and update command line arguments
        loaded_config = YAMLNamespace(configurations[args.config])
        args = parser.parse_args(namespace=loaded_config)
    return args


def get_common_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for training')
    parser.add_argument('--model', choices=models.available_models.keys(),  default='resnet18', help="Choose model")
    parser.add_argument('--pipeline-splits', type=str, nargs='+', default=[], help="List of the splitting layers")
    parser.add_argument('--replicas', type=int, default=1, help="Number of IPU replicas")
    parser.add_argument('--device-iterations', type=int, default=1, help="Device Iteration")
    parser.add_argument('--half-partial', action='store_true', help='Accumulate matrix multiplication partials in half precision')
    parser.add_argument('--norm-type', choices=['batch', 'group', 'none'], default='batch',  help="Set normalization layers in the model")
    parser.add_argument('--norm-num-groups', type=int, default=32, help="In case of group normalization, the number of groups")
    return parser
