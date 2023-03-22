# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.python import ipu
from time import perf_counter

logger = logging.getLogger(__name__)


def configure_ipu(args):
    # Standard IPU TensorFlow setup.
    ipu_config = ipu.config.IPUConfig()
    ipu_config.compilation_poplar_options = {"opt.internalExchangeOptimisationTarget": args.internal_exchange_target}

    # Enable float exception and stochastic rounding
    ipu_config.floating_point_behaviour.set_all = True

    ipu_config.allow_recompute = True

    # Set matmul options, including the precision of the "partials" (float or half)
    # and the "available memory proportion", which defines how much memory the
    # convolution planner can use for temporary values.
    # Both of these can be used to reduce memory usage in the model,
    # potentially with impact on accuracy and throughput respectively.
    matmul_options = {"partialsType": args.partials_type}
    if len(args.available_memory_proportion) == 1:
        matmul_options["availableMemoryProportion"] = str(args.available_memory_proportion[0])
    ipu_config.matmuls.poplar_options = matmul_options

    # Set convolution options, including the "partials" and "available memory proportion"
    # as in the above matmul options.
    conv_options = matmul_options.copy()
    ipu_config.convolutions.poplar_options = conv_options

    # Attach to given number of IPUs
    ipu_config.auto_select_ipus = args.nb_ipus_per_replica * args.replicas
    ipu_config.configure_ipu_system()


class PerfCallback(keras.callbacks.Callback):
    """Each epoch, capture execution time to compute throughput"""

    def __init__(self, steps_per_execution, batch_size):
        self.samples_per_execution = steps_per_execution * batch_size

    def on_epoch_begin(self, epoch, logs=None):
        t0 = perf_counter()
        self.t0 = t0

    def on_epoch_end(self, epoch, logs=None):
        t0 = self.t0
        t1 = perf_counter()
        d = t1 - t0
        tput = "{0:.15f}".format(self.samples_per_execution / d)
        logger.info(f"Execution {epoch}.\t Time: {d} seconds\t throughput: {tput} samples/sec.")


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    ipu.utils.reset_ipu_seed(seed)


def pretty_print_nested_list(a):
    def pretty_print_list(a):
        return ", ".join([f"{b:.5f}" for b in a])

    return "\n ".join([f"{pretty_print_list(b)}" for b in a])


def setup_logger(log_level):
    # Define a root config with a format which is simpler for console use
    logging.basicConfig(
        level=log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Define a specific Handler for this file that removes the root name.
    console = logging.StreamHandler()
    console.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
