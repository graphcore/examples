# Copyright 2020 Graphcore Ltd.
import logging
import sys


def get_basic_logger(name):
    lh = logging.StreamHandler(sys.stdout)
    lh.setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[lh])
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
