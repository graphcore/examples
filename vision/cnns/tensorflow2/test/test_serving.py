# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
import pytest

from examples_tests.test_util import SubProcessChecker
from test_common import run_serving


@pytest.mark.skip(reason='use of external data (T68092)')
class Simple(SubProcessChecker):
    def test_simple_serving_bs_1(self):
        path_to_imagenet = '/localdata/datasets/imagenet-data/'
        if not os.path.exists(path_to_imagenet):
            raise NameError(
                f'Directory {path_to_imagenet} from TFDS should have been copied to CI for this test')

        run_serving(self, '--config', 'resnet50_infer_bs1',
                    '--dataset-path', path_to_imagenet,
                    '--batch-size', '1',
                    '--port', '8502',
                    '--num-threads', '1',
                    '--verbose')
