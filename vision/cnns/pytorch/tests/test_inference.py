# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import pytest
import poptorch
import torch
import torchvision
from PIL import Image
from pathlib import Path
import import_helper
import models
import datasets
from utils import get_max_thoughput, run_script


class TestInference:
    @pytest.mark.ipus(2)
    def test_replicate(self):
        out = run_script("inference/run_benchmark.py", "--data synthetic --replicas 2 --model resnet18 --precision 16.16 --iterations 10 --dataloader-worker 4")
        max_throughput = get_max_thoughput(out)
        assert max_throughput > 0

    @pytest.mark.ipus(2)
    def test_syntetic_pipeline(self):
        out = run_script("inference/run_benchmark.py", "--data synthetic --batch-size 2 --model resnet18 --pipeline-splits layer3/0 --device-iterations 4 --precision 16.16 --iterations 10 --dataloader-worker 4")
        max_throughput = get_max_thoughput(out)
        assert max_throughput > 0

    @pytest.mark.ipus(2)
    def test_realdata_pipeline(self):
        out = run_script("inference/run_benchmark.py", "--data real --batch-size 2 --model resnet18 --pipeline-splits layer3/0 --device-iterations 4 --precision 16.16 --iterations 10 --dataloader-worker 4")
        max_throughput = get_max_thoughput(out)
        assert max_throughput > 0

    @pytest.mark.ipus(1)
    def test_full_precision(self):
        out = run_script("inference/run_benchmark.py", f"--data synthetic --model resnet18 --batch-size 1 --precision 32.32 --iterations 10 --dataloader-worker 4")
        max_thoughput = get_max_thoughput(out)
        assert max_thoughput > 0

    @pytest.mark.ipus(1)
    def test_IO_overlap(self):
        out = run_script("inference/run_benchmark.py", f"--config resnet50 --data generated --replicas 1 --batch-size 1 --iterations 10 --num-io-tiles 32")
        max_thoughput = get_max_thoughput(out)
        assert max_thoughput > 0


@pytest.mark.ipus(1)
@pytest.mark.parametrize("model_name", ["resnext50", "mobilenet"])
def test_single_ipu_models(model_name):
    out = run_script("inference/run_benchmark.py", f"--data synthetic --batch-size 1 --model {model_name} --iterations 10 --precision 16.16 --dataloader-worker 4")
    max_throughput = get_max_thoughput(out)
    assert max_throughput > 0


@pytest.mark.ipus(1)
@pytest.mark.parametrize("norm_layer", ["group", "none"])
def test_normlayer_resnet(norm_layer):
    out = run_script("inference/run_benchmark.py", f"--data synthetic --batch-size 1 --model resnet18 --norm-type {norm_layer} --iterations 10 --precision 16.16 --dataloader-worker 4")
    max_thoughput = get_max_thoughput(out)
    assert max_thoughput > 0


@pytest.mark.ipus(1)
@pytest.mark.parametrize("norm_layer", ["group", "none"])
def test_normlayer_efficientnet(norm_layer):
    out = run_script("inference/run_benchmark.py", f"--data synthetic --batch-size 1 --model efficientnet-b0 --norm-type {norm_layer} --norm-num-groups 4 --iterations 10 --precision 16.16 --dataloader-worker 4 --random-weights")
    max_thoughput = get_max_thoughput(out)
    assert max_thoughput > 0


@pytest.mark.ipus(1)
@pytest.mark.parametrize("precision", ["16.16", "32.32"])
@pytest.mark.parametrize("model_name", ["resnet18", "resnet50", "efficientnet-b0", pytest.param("efficientnet-b4", marks=pytest.mark.ipu_version("ipu2"))])
def test_pretrained_prediction(precision, model_name):
    ground_truth = [('zebra.jpg', 340), ('jay.jpg', 17), ('polar_bear.jpg', 296), ('banana.jpg', 954),
                    ('hippo.jpg', 344), ('ostrich.jpg', 9), ('ping-pong_ball.jpg', 722), ('pelican.jpg', 144)]

    class HelperClass:
        def __init__(self):
            pass
    args = HelperClass()
    args.model = model_name
    args.data = "imagenet"
    args.norm_type = "batch"
    args.norm_eps = 1e-5
    args.batchnorm_momentum = 0.1
    args.pipeline_splits = []
    args.normalization_location = "host"
    args.precision = precision
    args.efficientnet_expand_ratio = 6
    args.efficientnet_group_dim = 1
    args.num_io_tiles = 0
    model = models.get_model(args, datasets.datasets_info[args.data], pretrained=True)
    model.eval()
    opts = poptorch.Options()
    if precision == "16.16":
        opts.Precision.setPartialsType(torch.float16)
    else:
        opts.Precision.setPartialsType(torch.float32)

    poptorch_model = poptorch.inferenceModel(model, opts)

    input_size = models.model_input_shape(args)[1]
    augment = datasets.get_preprocessing_pipeline(train=False, half_precision=True if precision == "16.16" else False, input_size=input_size)
    for img_name, class_id in ground_truth:
        sample = augment(Image.open(os.path.join(Path(__file__).parent.parent.absolute(), "data/images/", img_name))).view(1, 3, input_size, input_size)
        pred = poptorch_model(sample)
        assert class_id == torch.argmax(pred), f"Prediction for {img_name} was incorrect."


@pytest.mark.ipus(1)
def test_pretrained_batchnorm_fp16():
    fake_data = torch.ones(1, 64, 10, 10)
    model = torchvision.models.resnet18(pretrained=True).bn1   # Get a batchnorm layer from a real world model.
    cpu_mean = model.running_mean
    cpu_var = model.running_var
    model.half()
    opts = poptorch.Options()
    opts.anchorTensor('running_mean', 'running_mean')
    opts.anchorTensor('running_var', 'running_var')
    opts.Precision.runningStatisticsAlwaysFloat(True)

    poptorch_model = poptorch.inferenceModel(model, opts)
    output = poptorch_model(fake_data)   # Compile the model.
    ipu_mean = poptorch_model.getAnchoredTensor('running_mean')
    ipu_var = poptorch_model.getAnchoredTensor('running_var')

    assert torch.allclose(ipu_mean, cpu_mean.half())
    assert torch.allclose(ipu_var, cpu_var.half())
