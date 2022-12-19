# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
import pytest
from pathlib import Path
from examples_tests.test_util import SubProcessChecker
from test_common import run_serving


@pytest.mark.skip(reason='serving bin not in CI (T67924)')
class Simple(SubProcessChecker):
    def test_simple_serving_bs_1(self):
        path_to_dataset = Path(__file__).absolute().parent.parent
        if not os.path.exists(path_to_dataset):
            raise NameError(
                f'Directory {path_to_dataset} from TFDS should have been copied to CI for this test')

        run_serving(self, '--config', 'resnet50_infer_test',
                    '--dataset-path', path_to_dataset,
                    '--batch-size', '1',
                    '--port', '8502',
                    '--num-threads', '1',
                    '--verbose')
