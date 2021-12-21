# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
Tests for the embedded runtime inference script.
"""

import os
import pytest
from pathlib import Path
from examples_tests.test_util import SubProcessChecker


script_dir = Path(__file__).parent.parent


class InferenceEmbeddedTest(SubProcessChecker):

    def _run_test_inference_embedded(self, cmd_args, extra_env={}):
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        env.update(extra_env)
        self.run_command(cmd_args, script_dir, ["Latencies - avg"], env=env)

    @pytest.mark.ipus(1)
    def test_embedded_inference_functionality(self):
        # Basic functionality test with generated data and small test config
        self._run_test_inference_embedded("python3 inference_embedded.py --model resnet --config mk2_resnet8_test --generated-data --iterations 10 --batches-per-step 10 --micro-batch-size 1 --eight-bit-io --force-recompile")
