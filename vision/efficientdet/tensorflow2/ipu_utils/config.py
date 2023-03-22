# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import tensorflow as tf
from absl import logging
from tensorflow.python import ipu

from . import set_or_add_env


def ipu_configure(args: argparse.Namespace):
    logging.info("Configuring IPU...")
    partials_type = "half" if args.partials_precision == tf.float16 else "float"

    cfg = ipu.config.IPUConfig()
    cfg.experimental.always_rearrange_copies_on_the_host = not args.opt_device_rearrange
    cfg.optimizations.prefetch_data_streams = args.opt_prefetch_data_streams

    if args.opt_liveness_scheduler != ipu.config.SchedulingAlgorithm.CHOOSE_BEST:
        cfg.scheduling.algorithm = args.opt_liveness_scheduler
        if args.opt_liveness_scheduler == ipu.config.SchedulingAlgorithm.LOOK_AHEAD:
            cfg.scheduling.maximum_scheduler_lookahead_depth = args.opt_scheduler_lookahead_depth
            cfg.scheduling.maximum_scheduler_search_space_size = args.opt_scheduler_lookahead_search_space

    cfg.device_connection.version = "ipu2" if args.ipu_connection_type == ipu.config.DeviceConnectionType.NEVER else ""
    cfg.device_connection.type = args.ipu_connection_type

    cfg.convolutions.poplar_options = {
        "partialsType": partials_type,
        "enableConvDithering": str(args.opt_conv_dithering).lower(),
        "availableMemoryProportion": str(args.available_memory_proportion),
    }

    cfg.matmuls.poplar_options = {
        "partialsType": partials_type,
        "availableMemoryProportion": str(args.available_memory_proportion),
    }

    cfg.auto_select_ipus = 1

    if args.opt_use_io_tiles:
        cfg.io_tiles.num_io_tiles = args.opt_num_io_tiles
        cfg.io_tiles.place_ops_on_io_tiles = True

    logging.debug(cfg)
    cfg.configure_ipu_system()
    logging.info("Configured")


def ipu_engine_options(args: argparse.ArgumentParser):
    engine_opts = {}
    if args.profile_dir is not None:
        os.makedirs(args.profile_dir, exist_ok=True)

        get_execution_report = str(not args.skip_execution_report).lower()

        engine_opts = {
            **engine_opts,
            **{
                "autoReport.all": "true",
                "autoReport.directory": args.profile_dir,
                "debug.instrument": get_execution_report,
            },
        }

    if args.opt_internal_exchange_target is not None:
        engine_opts["opt.internalExchangeOptimisationTarget"] = args.opt_internal_exchange_target

    engine_opts["streamCallbacks.numWorkerThreads"] = "6"
    engine_opts["streamCallbacks.multiThreadMode"] = "collaborative"
    engine_opts["streamCallbacks.maxLookahead"] = "unlimited"

    if len(engine_opts) > 0:
        logging.debug(engine_opts)
        set_or_add_env("POPLAR_ENGINE_OPTIONS", engine_opts)
