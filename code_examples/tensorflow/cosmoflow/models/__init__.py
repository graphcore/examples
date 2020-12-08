# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

"""
Keras example model factory functions.
"""

import importlib


def get_model(name, **model_args):
    """Factory function for constructing a model by name with args"""
    module = importlib.import_module('.' + name, 'models')
    return module.build_model(**model_args)
