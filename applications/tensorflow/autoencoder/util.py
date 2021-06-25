# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.config import IPUConfig


def get_config(opts, training=True, profiling=False):
    """Builds ipu_options
    """
    config = IPUConfig()

    ipus = opts.select_ipus
    if ipus[0] == -1:
        train_ipus = 1  # opts.shards
        valid_ipus = 1  # This might want an option to control
        if not opts.multiprocessing:
            config.auto_select_ipus = [train_ipus, valid_ipus]
        else:
            ipus = train_ipus if training else valid_ipus
            config.auto_select_ipus = [ipus]
    else:
        if opts.multiprocessing:
            ipus = [ipus[0] if training else ipus[1]]
        config.select_ipus = ipus

    config.floating_point_behaviour.esr = opts.prng

    return config
