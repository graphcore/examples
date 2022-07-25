# Copyright (c) 202 Graphcore Ltd. All rights reserved.
import sys
import pytest
from pathlib import Path
from typing import Callable

from examples_tests.test_util import SubProcessChecker

sys.path.append(str(Path(__file__).absolute().parent.parent))
from test_common import run_train


class CompilationOOM(SubProcessChecker):
    error_message: str = 'graph_memory_allocation_error'

    poprun_prefix: list = [
        'poprun',
        '--only-output-from-instance', '0'
    ]

    get_config: Callable = lambda self, config_name: (
        '--config', config_name,
        '--compile-only',
        '--num-epochs', '4',
        '--validation', 'False'
    )

    @pytest.mark.long_test
    def test_nopoprun_rn50_mlperf_config(self):
        self.assertNotIn(
            run_train(self, *self.get_config('resnet50_mlperf_bs16')),
            self.error_message
        )

    @pytest.mark.long_test
    def test_nopoprun_rn50_config(self):
        self.assertNotIn(
            run_train(self, *self.get_config('resnet50_16ipus_8k_bn_pipeline')),
            self.error_message
        )

    @pytest.mark.long_test
    def test_poprun_rn50_mlperf_pod16_config(self):
        poprun_prefix = self.poprun_prefix + [
            '--num-instances', '8',
            '--num-replicas', '16'
        ]
        self.assertNotIn(
            run_train(self, *self.get_config('resnet50_mlperf_bs16'), poprun_prefix=poprun_prefix),
            self.error_message)

    @pytest.mark.long_test
    def test_poprun_rn50_config(self):
        poprun_prefix = self.poprun_prefix + [
            '--ipus-per-replica', '4',
            '--num-instances', '2',
            '--num-replicas', '2'
        ]
        self.assertNotIn(
            run_train(self, *self.get_config('resnet50_16ipus_8k_bn_pipeline'), poprun_prefix=poprun_prefix),
            self.error_message
        )
