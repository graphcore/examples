# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from absl import flags

flags.DEFINE_float('available_memory_proportion', 0.2, lower_bound=0., upper_bound=1.,
                   help='memory proportion to reserve for matmuls')

FLAGS = flags.FLAGS

"""
A portable API for IPU or non-IPU code.

Note: Assumes you'll use the IPU whenever gc-tensorflow is installed.
"""

try:
    from tensorflow.python import ipu

    IS_IPU = True
except ImportError:
    import tensorflow as tf

    IS_IPU = False

if IS_IPU:
    gather = ipu.embedding_ops.embedding_lookup
    from tensorflow.python.ipu.keras.layers import Dropout, LayerNormalization

else:
    gather = tf.gather
    from tensorflow.keras.layers import Dropout, LayerNormalization


class DummyScope:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DummyStrategy:
    def __init__(self):
        self.scope = DummyScope


def configure_and_get_strategy():
    if IS_IPU:
        config = ipu.config.IPUConfig()
        config.auto_select_ipus = FLAGS.num_ipus
        config.floating_point_behaviour.esr = True
        config.allow_recompute = False
        config.matmuls.poplar_options['partialsType'] = 'half'
        config.matmuls.poplar_options["availableMemoryProportion"] = str(FLAGS.available_memory_proportion)
        config.compilation_poplar_options['opt.internalExchangeOptimisationTarget'] = 'balanced'
        ipu.utils.configure_ipu_system(config)
        strategy = ipu.ipu_strategy.IPUStrategy()
    else:
        strategy = DummyStrategy()

    return strategy
