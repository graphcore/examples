# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import logging
import sys
import os
import popdist


def get_basic_logger(name):
    log_levels_map = dict(
        CRITICAL = logging.CRITICAL,
        ERROR = logging.ERROR,
        WARNING = logging.WARNING,
        INFO = logging.INFO,
        DEBUG = logging.DEBUG,
        NOTSET = logging.NOTSET
    )
    log_level_env = os.getenv("RNNT_LOG_LEVEL")
    log_level = log_levels_map.get(log_level_env, logging.INFO)

    lh = logging.StreamHandler(sys.stdout)
    lh.setLevel(log_level)
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(module)s - %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[lh])
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if popdist.isPopdistEnvSet():
        instance_idx = popdist.getInstanceIndex()
    else:
        instance_idx = 0
    if instance_idx > 0:
        # to avoid excess logging, disabling logging for instance_idxs > 0
        logger.disabled = True

    return logger
