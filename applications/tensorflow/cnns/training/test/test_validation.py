# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import unittest
import pytest
from tempfile import TemporaryDirectory
from pathlib import Path
from examples_tests.test_util import SubProcessChecker
import sys

sys.path.append(str(Path(__file__).absolute().parent.parent))

from test_common import run_train, cifar10_data_dir, run_validation

working_path = Path(__file__).parent.parent


class TestBasicFunctionality(SubProcessChecker):
    """ Test that the help option works"""
    def test_help(self):
        help_out = run_validation(working_path, **{'--help': ''})
        self.assertNotEqual(help_out.find("usage: validation.py"), -1)


@pytest.mark.ipus(1)
class TestCifar10Validation(SubProcessChecker):
    """ Check validation for cifar-10."""

    def test_cifar10_validation(self):
        with TemporaryDirectory() as log_dir:
            # create checkpoints for iterations 0, 2 and 4
            out = run_train(
                self,
                **{
                    '--data-dir': cifar10_data_dir,
                    '--name-suffix': 'test_validation',
                    '--log-dir': log_dir,
                    '--iterations': 4,
                    '--batches-per-step': 2,
                    '--no-validation': '',
                    '--ckpts-per-epoch': 1000000})
            dir_to_restore = None
            for line in out.split('\n'):
                if line.find('Saving to ') != -1:
                    dir_to_restore = line[11:]
                    break
            self.assertIsNotNone(dir_to_restore)

            # run validation on these checkpoints
            out = run_validation(working_path,
                                 **{'--data-dir': cifar10_data_dir,
                                    '--restore-path': dir_to_restore})
            validation_line_count = 0
            iterations_in_out = [0, 2, 4]
            for line in out.split('\n'):
                prefix = 'iteration:'
                pos = line.find(prefix)
                if pos != -1:
                    iteration = int(line[pos + len(prefix):line.find(',', pos)])
                    self.assertEqual(iterations_in_out[validation_line_count], iteration)
                    validation_line_count += 1
            self.assertEqual(3, validation_line_count)
