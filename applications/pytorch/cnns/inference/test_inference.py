# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import subprocess
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
        return float(output[pos+4:].split(',')[0][:-7])
    else:
        return -1


class TestInference:
    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_real_data(self):
        out = run_inference("--data real --model resnet18 --precision 16.16 --iterations 10")
        max_throughput = get_max_thoughput(out)
        assert max_throughput > 0

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_replicate(self):
        out = run_inference("--data synthetic --replicas 2 --model resnet18 --precision 16.16 --iterations 10")
        max_throughput = get_max_thoughput(out)
        assert max_throughput > 0

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_syntetic_pipeline(self):
        out = run_inference("--data synthetic --batch-size 2 --model resnet18 --pipeline-splits layer3/0 --device-iterations 4 --precision 16.16 --iterations 10")
        max_throughput = get_max_thoughput(out)
        assert max_throughput > 0

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_realdata_pipeline(self):
        out = run_inference("--data real --batch-size 2 --model resnet18 --pipeline-splits layer3/0 --device-iterations 4 --precision 16.16 --iterations 10")
        max_throughput = get_max_thoughput(out)
        assert max_throughput > 0


    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_full_precision(precision):
        out = run_inference(f"--data synthetic --batch-size 1 --precision 32.32 --iterations 10")
        max_thoughput = get_max_thoughput(out)
        assert max_thoughput > 0


@pytest.mark.category2
@pytest.mark.ipus(1)
@pytest.mark.parametrize("model_name", ["resnet34", "resnet50", "resnext50", "mobilenet", "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4"])
def test_single_ipu_models(model_name):
    out = run_inference(f"--data synthetic --batch-size 1 --model {model_name} --iterations 10 --precision 16.16")
    max_throughput = get_max_thoughput(out)
    assert max_throughput > 0


@pytest.mark.category2
@pytest.mark.ipus(1)
@pytest.mark.parametrize("norm_layer", ["group", "none"])
def test_normlayer_resnet(norm_layer):
    out = run_inference(f"--data synthetic --batch-size 1 --model resnet18 --norm-type {norm_layer} --iterations 10 --precision 16.16")
    max_thoughput = get_max_thoughput(out)
    assert max_thoughput > 0


@pytest.mark.category2
@pytest.mark.ipus(1)
@pytest.mark.parametrize("norm_layer", ["group", "none"])
def test_normlayer_efficientnet(norm_layer):
    out = run_inference(f"--data synthetic --batch-size 1 --model efficientnet-b0 --norm-type {norm_layer} --norm-num-groups 4 --iterations 10 --precision 16.16")
    max_thoughput = get_max_thoughput(out)
    assert max_thoughput > 0
