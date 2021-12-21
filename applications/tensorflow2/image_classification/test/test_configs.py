# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import unittest
import argparse

from configuration import file_argparse, terminal_argparse


class LoadingConfigsTest(unittest.TestCase):

    def test_loading_configs(self):
        parser = argparse.ArgumentParser(description='Test configs',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = terminal_argparse.add_arguments(parser)
        args = parser.parse_args(['--config', 'resnet8_test'])
        args = file_argparse.parse_yaml_config(args, parser)

        assert args.model_name == 'cifar_resnet8'
