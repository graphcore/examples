# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest
from tutorials_tests.testing_util import run_command_fail_explicitly


cwd = Path(__file__).parent.parent


class TestInference:
    @pytest.mark.category1
    @pytest.mark.ipus(2)
    def test_inference(self):
        cmd = ["python", "inference_pipelining.py"]
        out = run_command_fail_explicitly(cmd, cwd)

    @pytest.mark.category1
    @pytest.mark.ipus(2)
    def test_inference_sequential(self):
        cmd = ["python", "inference_pipelining_sequential.py"]
        out = run_command_fail_explicitly(cmd, cwd)

    @pytest.mark.category1
    @pytest.mark.ipus(2)
    def test_inference_set_pipeline_stage(self):
        cmd = ["python", "inference_pipelining_set_pipeline_stage.py"]
        out = run_command_fail_explicitly(cmd, cwd)


class TestTraining:
    @pytest.mark.category1
    @pytest.mark.ipus(4)
    def test_training(self):
        cmd = ["python", "training_pipelining.py"]
        out = run_command_fail_explicitly(cmd, cwd)

    @pytest.mark.category1
    @pytest.mark.ipus(4)
    def test_training_sequential(self):
        cmd = ["python", "training_pipelining_sequential.py"]
        out = run_command_fail_explicitly(cmd, cwd)
