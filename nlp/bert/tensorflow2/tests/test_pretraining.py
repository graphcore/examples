# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os

from examples_tests.test_util import SubProcessChecker
from tests.utils import get_app_root_dir


class TestBuildAndRun(SubProcessChecker):
    @staticmethod
    def _get_sample_dataset_path():
        app_dir = get_app_root_dir()
        return app_dir.joinpath("data_utils").joinpath("wikipedia").resolve()

    @staticmethod
    def _get_pretraining_command(extra_args=None):
        base_cmd = "python run_pretraining.py --config tests/pretrain_tiny_test.json"
        if extra_args is not None and len(extra_args) > 0:
            base_cmd += " " + " ".join(extra_args)
        print(f"Running: {base_cmd}")
        return base_cmd


class TestRunPretraining(TestBuildAndRun):
    def test_run_pretraining(self):
        cmd = self._get_pretraining_command(extra_args=["--dataset-dir", str(self._get_sample_dataset_path())])
        self.run_command(cmd, get_app_root_dir(), [""])

    def test_run_pretraining_compile_only(self):
        cmd = self._get_pretraining_command(
            extra_args=["--dataset-dir", str(self._get_sample_dataset_path()), "--compile-only"]
        )
        env = os.environ.copy()
        env["TF_POPLAR_FLAGS"] = "--executable_cache_path=./" + env.get("TF_POPLAR_FLAGS", "")
        self.run_command(cmd, get_app_root_dir(), [""], env=env)
