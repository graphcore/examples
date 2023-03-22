# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest

from tests.test_pretraining import TestBuildAndRun
from tests.utils import get_app_root_dir


class TestDistributedRunPretraining(TestBuildAndRun):
    @pytest.mark.skip
    def test_distributed_training(self):
        cmd = self._get_pretraining_command(extra_args=["--dataset-dir", str(self._get_sample_dataset_path())])
        poprun_prefix = "poprun --only-output-from-instance 0 --num-instances 2 --num-replicas 2 --ipus-per-replica 2"
        poprun_cmd = poprun_prefix + " " + cmd
        self.run_command(poprun_cmd, get_app_root_dir(), [""])
