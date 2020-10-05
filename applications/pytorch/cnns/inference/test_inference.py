# Copyright 2020 Graphcore Ltd.
import os
import subprocess
import unittest
import pytest


def run_inference(parameters):
    cwd = os.path.dirname(os.path.abspath(__file__))
    param_list = parameters.split(" ")
    cmd = ["python3", 'run_benchmark.py'] + param_list
    out = subprocess.check_output(cmd, cwd=cwd).decode("utf-8")
    return out


def get_max_thoughput(output):
    if output.find("Throughput at") != -1:
        pos = output.find("max=")
        return float(output[pos+4:].split(',')[0])


class TestInference(unittest.TestCase):
    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_real_data(self):
        out = run_inference("--data real --model resnet18")
        assert out.find("Throughput at") != -1

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_replicas(self):
        out = run_inference("--data synthetic --replicas 2 --model resnet18 --batch-size 1")
        assert out.find("Throughput at") != -1

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_syntetic_pipeline(self):
        out = run_inference("--data synthetic --batch-size 1 --model resnet50 --pipeline-splits layer3/2 --device-iteration 8")
        assert out.find("Throughput at") != -1

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_realdata_pipeline(self):
        out = run_inference("--data real --batch-size 1 --model resnet50 --pipeline-splits layer3/2 --device-iteration 8")
        assert out.find("Throughput at") != -1


@pytest.mark.category2
@pytest.mark.ipus(1)
@pytest.mark.parametrize("model_name", ["resnet18", "resnet34", "efficientnet-b0"])
def test_single_ipu_models(model_name):
    out = run_inference("--data synthetic --batch-size 1 --model {}".format(model_name))
    assert out.find("Throughput at") != -1
