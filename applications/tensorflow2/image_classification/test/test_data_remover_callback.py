# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import unittest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))
from callbacks.data_remover_callback import DataRemoverCallback


class TestDataRemoverCallback(unittest.TestCase):

    def setUp(self):
        self.callback_instance = DataRemoverCallback(['one_field', 'another_field'])

    def test_remove_one_field_on_train_batch_end(self):
        logs = {'one_field': 1}
        self.callback_instance.on_train_batch_end(0, logs)
        self.assertEqual(logs, {})

    def test_remove_one_field_on_test_batch_end(self):
        logs = {'one_field': 1}
        self.callback_instance.on_test_batch_end(0, logs)
        self.assertEqual(logs, {})

    def test_remove_multiple_fields(self):
        logs = {'one_field': 1, 'another_field': 2}
        self.callback_instance.on_train_batch_end(0, logs)
        self.assertEqual(logs, {})

    def test_data_not_removed(self):
        logs = {'a': 1, 'b': 2, 'one_field': 3}
        self.callback_instance.on_train_batch_end(0, logs)
        self.assertEqual(logs, {'a': 1, 'b': 2})
