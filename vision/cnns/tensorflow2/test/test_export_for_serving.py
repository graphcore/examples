# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
import tempfile
import pytest

from test_common import run_export
from examples_tests.test_util import SubProcessChecker
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import signature_constants

@pytest.mark.skip(reason='use of external data (T68092)')
class Simple(SubProcessChecker):
    def test_simple_export_bs_1(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')

        with tempfile.TemporaryDirectory() as tmp_folder:
            run_export(self, '--export-dir', tmp_folder,
                       '--dataset-path', '/localdata/datasets/',
                       '--micro-batch-size', '1')
            imported = load.load(tmp_folder)
            loaded = imported.signatures[
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

            inputs = [input for input in loaded.inputs if "input_" in input.name]
            self.assertEqual(inputs[0].shape, (1, 32, 32, 3))

    def test_simple_export_bs_16(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')

        with tempfile.TemporaryDirectory() as tmp_folder:
            run_export(self, '--export-dir', tmp_folder,
                       '--dataset-path', '/localdata/datasets/',
                       '--micro-batch-size', '16')
            imported = load.load(tmp_folder)
            loaded = imported.signatures[
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

            inputs = [input for input in loaded.inputs if "input_" in input.name]
            self.assertEqual(inputs[0].shape, (16, 32, 32, 3))
