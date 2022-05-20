# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import unittest
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

from configuration import file_argparse, terminal_argparse


class LoadingConfigsTest(unittest.TestCase):

    def test_loading_configs(self):
        # Non void argv is passed to argparse with pytest so we need to make it empty
        argv_bc = sys.argv

        sys.argv = []

        parser = argparse.ArgumentParser(prog='test_configs.py', description='Test configs',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = terminal_argparse.add_arguments(parser)
        args = parser.parse_args(['--config', 'resnet8_test'])
        args = file_argparse.parse_yaml_config(args, parser)

        sys.argv = argv_bc

        assert args.model_name == 'cifar_resnet8'
