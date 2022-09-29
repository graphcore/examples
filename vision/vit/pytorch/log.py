# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

import logging
import sys
import datetime
from logging import handlers


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    # Predefined log level includes, from highest to lowest severity:
    # CRITICAL, ERROR, WARNING, INFO, DEBUG

    def __init__(self, level='INFO', when='D', backCount=3,
                 fmt='[%(asctime)s] %(message)s'):
        self.logger = logging.getLogger()
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(logging.getLevelName(level))
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(format_str)
        self.logger.addHandler(sh)


logger = Logger(level='INFO').logger
