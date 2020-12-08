# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

"""
Keras dataset specifications.
"""


def get_datasets(name, **data_args):
    if name == 'dummy':
        from .dummy import get_datasets
        return get_datasets(**data_args)
    elif name == 'cosmo':
        from .cosmo import get_datasets
        return get_datasets(**data_args)
    else:
        raise ValueError('Dataset %s unknown' % name)
