# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from absl import flags

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

from tensorflow import gather

if IS_IPU:
    from ipu_tensorflow_addons.keras.layers import Dropout, LayerNormalization
    from ipu_tensorflow_addons.keras.optimizers import AdamIpuOptimizer

    from static_ops.static_ops import grouped_gather, grouped_scatter
else:
    from tensorflow.keras.layers import Dropout, LayerNormalization
    from tensorflow.keras.optimizers import Adam

    from layers import _scatter

    # uses list comprehension over the first dimension
    # note: these are implemented for cross-compatibility and are slow
    def grouped_scatter(data: tf.Tensor, indices: tf.Tensor, table_size: int) -> tf.Tensor:
        return _scatter(data, indices, table_size, gather_scatter_method="debug")

    def grouped_gather(params: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
        return gather(params, indices, batch_dims=1)


flags.DEFINE_float(
    "available_memory_proportion",
    0.2,
    lower_bound=0.0,
    upper_bound=1.0,
    help="memory proportion to reserve for matmuls",
)
flags.DEFINE_enum(
    "optimization_target", "cycles", ("balanced", "cycles", "memory"), help="optimization target for the planner"
)
flags.DEFINE_enum(
    "scheduling_algorithm",
    "CHOOSE_BEST",
    ("CHOOSE_BEST", "SHORTEST_PATH", "CLUSTERING"),
    "the schedling algorithm to use.",
)
flags.DEFINE_integer("maximum_cross_replica_sum_buffer_size", 1000000, "max size of the cross-replica sum buffer")

FLAGS = flags.FLAGS


class DummyScope:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DummyStrategy:
    def __init__(self):
        self.scope = DummyScope


def configure_and_get_strategy(num_replicas):
    if IS_IPU:
        config = ipu.config.IPUConfig()
        config.auto_select_ipus = num_replicas
        config.floating_point_behaviour.esr = ipu.config.StochasticRoundingBehaviour.ON
        config.matmuls.poplar_options["partialsType"] = "half"
        config.matmuls.poplar_options["availableMemoryProportion"] = str(FLAGS.available_memory_proportion)
        config.slices.poplar_options["availableMemoryProportion"] = str(FLAGS.available_memory_proportion)
        # balanced, memory or cycles
        config.compilation_poplar_options["opt.internalExchangeOptimisationTarget"] = FLAGS.optimization_target
        config.scheduling.algorithm = vars(ipu.config.SchedulingAlgorithm)[FLAGS.scheduling_algorithm]
        config.optimizations.maximum_cross_replica_sum_buffer_size = FLAGS.maximum_cross_replica_sum_buffer_size
        ipu.utils.configure_ipu_system(config)
        strategy = ipu.ipu_strategy.IPUStrategy()
    else:
        strategy = DummyStrategy()

    return strategy
