# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

"""Utility code for argparse"""

import argparse
import yaml


class ReadYaml(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = yaml.load(values, Loader=yaml.FullLoader)
        setattr(namespace, self.dest, my_dict)
