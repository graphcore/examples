# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from tensorflow.python.ipu import utils


def get_config(opts, training=True, profiling=False):
    """Builds ipu_options
    """
    config = utils.create_ipu_config(profiling=profiling)

    ipus = opts.select_ipus
    if ipus[0] == -1:
        train_ipus = 1  # opts.shards
        valid_ipus = 1  # This might want an option to control
        if not opts.multiprocessing:
            config = utils.auto_select_ipus(config, [train_ipus, valid_ipus])
        else:
            ipus = train_ipus if training else valid_ipus
            config = utils.auto_select_ipus(config, [ipus])
    else:
        if opts.multiprocessing:
            ipus = [ipus[0] if training else ipus[1]]
        config = utils.select_ipus(config, ipus)

    config = utils.set_compilation_options(config, {
        "prng.enable": "true" if opts.prng else "false"
    })

    return config
