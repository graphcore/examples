# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import yaml
import argparse



def add_arguments(parser, config_filename):
    configs = get_available_configs(config_filename)
    parser.add_argument('--config', choices=configs.keys(), help="Select from avalible configurations")
    return parser


def get_available_configs(config_filename):
    with config_filename.open() as config_file:
        configs = yaml.full_load(config_file)
    return configs


class YAMLNamespace(argparse.Namespace):
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def parse_config(args, parser, config_filename, known_args_only=False):
    if args.config is not None:
        # Load the configurations from the YAML file and update command line arguments
        configs = get_available_configs(config_filename)
        loaded_config = YAMLNamespace(configs[args.config])
        if known_args_only:
            config_args, _ = parser.parse_known_args(namespace=loaded_config)
            parser.set_defaults(**vars(config_args))
            cmdline_args, _ = parser.parse_known_args()
        else:
            config_args = parser.parse_args(namespace=loaded_config)
            parser.set_defaults(**vars(config_args))
            cmdline_args = parser.parse_args()
        args = cmdline_args

    return args
