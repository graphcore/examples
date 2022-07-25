# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import logging
import sys
from logging import handlers


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    # Predefined log level includes, from highest to lowest severity:
    # CRITICAL, ERROR, WARNING, INFO, DEBUG

    def __init__(self, filename=None, level='INFO', when='D', backCount=3,
                 fmt='[%(asctime)s] %(message)s'):
        assert filename is not None
        self.filename = filename
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(logging.getLevelName(level))
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when,
                                               backupCount=backCount, encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


if __name__ == '__main__':
    log = Logger('all.log', level='ERROR')
    log.logger.debug('debug')
    log.logger.info('info')
    log.logger.warning('warning')
    log.logger.error('error')
    log.logger.critical('critical')
