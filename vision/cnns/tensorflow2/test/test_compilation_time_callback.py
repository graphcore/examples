# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import unittest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))
from callbacks.compilation_time_callback import CompilationTimeCallback


class Mock_time:
    time_step = 10
    counter = 0

    @staticmethod
    def time():
        Mock_time.counter += Mock_time.time_step
        return Mock_time.counter


class CompilationTimeTest(unittest.TestCase):

    @unittest.mock.patch('callbacks.compilation_time_callback.time', Mock_time)
    def test_compilation_time(self):

        callback = CompilationTimeCallback()

        callback.on_train_begin()

        logs = {}
        callback.on_train_batch_begin(0, logs)
        callback.on_train_batch_end(0, logs)
        self.assertEqual(logs, {'Compilation Time': 10})

        logs = {}
        callback.on_train_batch_begin(1, logs)
        callback.on_train_batch_end(1, logs)
        self.assertEqual(logs, {})
