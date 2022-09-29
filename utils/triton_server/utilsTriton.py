# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import glob
import os
import pytest
import signal
import traceback


def IsModelPathValid(path_candidate):
    ts_model_configs = glob.glob(
        path_candidate + '/*/*.pbtxt')
    return len(ts_model_configs) > 0


def GetModelPath(config, optionsKey):
    optionsPath = config.getoption(optionsKey)
    invParDir = config.invocation_params.dir
    if optionsPath is not None:
        possibleModelPath = os.path.abspath(optionsPath)
        if not IsModelPathValid(possibleModelPath):
            optionsPath = str(invParDir) + "/" + optionsPath
        else:
            optionsPath = possibleModelPath

    if optionsPath is None or not IsModelPathValid(optionsPath):
        optionsPath = str(invParDir) + "/models"
        if not IsModelPathValid(optionsPath):
            optionsPath = os.path.abspath(config.args[0]) + "/models"
            if not IsModelPathValid(optionsPath):
                optionsPath = os.path.abspath(
                    config.args[0]) + "/tritonserver/models"
                if not IsModelPathValid(optionsPath):
                    optionsPath = str(invParDir) + "/" + \
                        config.args[0] + "/models"

    if not IsModelPathValid(optionsPath):
        pytest.fail("Model repo path: " +
                    optionsPath + " doesn't exist!")

    return optionsPath


class PoolLogExceptions:
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            return self.__callable(*args, **kwargs)
        # pylint: disable=W0703
        except Exception:
            pytest.fail(traceback.format_exc())


class Timeout:
    def __init__(self, seconds=1, process=None, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
        self.process = process

    def handle_timeout(self, *_):
        if self.process is not None:
            self.process.kill()
        pytest.fail(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, *_):
        signal.alarm(0)
