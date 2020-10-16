# Copyright 2020 Graphcore Ltd.
import os
import subprocess
import unittest
import pytest


def download_images():
    cwd = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(cwd, "images")):
        subprocess.check_output(["sh", "get_images.sh"], cwd=cwd)


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


class TestInference(unittest.TestCase):
    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_real_data(self):
        download_images()
        out = run_inference("--data real --model resnet18")
        max_throughput = get_max_thoughput(out)
        self.assertGreater(max_throughput, 0)

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_replicate(self):
        out = run_inference("--data synthetic --replicas 2 --model resnet18")
        max_throughput = get_max_thoughput(out)
        self.assertGreater(max_throughput, 0)


    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_syntetic_pipeline(self):
        out = run_inference("--data synthetic --batch-size 2 --model resnet50 --pipeline-splits layer3/2 --device-iteration 8")
        max_throughput = get_max_thoughput(out)
        self.assertGreater(max_throughput, 0)


    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_realdata_pipeline(self):
        download_images()
        out = run_inference("--data real --batch-size 2 --model resnet50 --pipeline-splits layer3/2 --device-iteration 8")
        max_throughput = get_max_thoughput(out)
        self.assertGreater(max_throughput, 0)

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_resnext101(self):
        out = run_inference("--data synthetic --batch-size 1 --model resnext101 --pipeline-splits layer3/6 layer3/20 layer4 --device-iteration 4")
        max_throughput = get_max_thoughput(out)
        self.assertGreater(max_throughput, 0)


@pytest.mark.category2
@pytest.mark.ipus(1)
@pytest.mark.parametrize("model_name", ["resnet18", "resnet34", "resnet50", "resnext50", "mobilenet", "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4"])
def test_single_ipu_models(model_name):
    out = run_inference("--data synthetic --batch-size 1 --model {}".format(model_name))
    max_throughput = get_max_thoughput(out)
    assert max_throughput > 0


@pytest.mark.category2
@pytest.mark.ipus(1)
@pytest.mark.parametrize("precision", ["half", "full"])
def test_precision(precision):
    out = run_inference(f"--data synthetic --batch-size 1 --precision {precision}")
    max_thoughput = get_max_thoughput(out)
    assert max_thoughput > 0


@pytest.mark.category2
@pytest.mark.ipus(1)
@pytest.mark.parametrize("norm_layer", ["group", "none"])
def test_normlayer_resnet(norm_layer):
    out = run_inference(f"--data synthetic --batch-size 1 --model resnet18 --normlayer {norm_layer}")
    max_thoughput = get_max_thoughput(out)
    assert max_thoughput > 0


@pytest.mark.category2
@pytest.mark.ipus(1)
@pytest.mark.parametrize("norm_layer", ["group", "none"])
def test_normlayer_efficientnet(norm_layer):
    out = run_inference(f"--data synthetic --batch-size 1 --model efficientnet-b0 --normlayer {norm_layer} --groupnorm-group-num 4")
    max_thoughput = get_max_thoughput(out)
    assert max_thoughput > 0
