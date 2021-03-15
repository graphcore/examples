# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import pytest
import os
import subprocess
import torch
import numpy as np
from torchvision import transforms
from datasets.preprocess import NormalizeToTensor, RandomResizedFlipCrop, ToHalf
import sys
sys.path.append('..')
import models


def run_script(script_name, parameters, python=True):
    cwd = os.path.dirname(os.path.abspath(__file__))
    param_list = parameters.split(" ")
    if python:
        cmd = ["python", script_name]
    else:
        cmd = [script_name]
    cmd = cmd + param_list
    out = subprocess.check_output(cmd, cwd=cwd).decode("utf-8")
    return out


class TestCustomAugmentations():
    @staticmethod
    def deterministic_get_params(*args):
        return 20, 20, 100, 100

    @pytest.mark.category1
    def test_custom_flip(self):
        img = torch.rand(1, 3, 100, 100)
        custom = RandomResizedFlipCrop.fast_hflip(img)
        correct = transforms.functional.hflip(img)
        assert torch.allclose(custom, correct, atol=1e-06)

    @pytest.mark.category1
    def test_NormalizeToTensor_pil(self):
        tensor = torch.rand(3, 100, 100)
        to_pil = transforms.ToPILImage()
        img = to_pil(tensor)
        custom_pipeline = NormalizeToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        correct_pipeline = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        custom = custom_pipeline(img)
        correct = correct_pipeline(img)
        assert torch.allclose(custom, correct, atol=1e-06)

    @pytest.mark.category1
    def test_RandomResizedFlipCrop_pil(self):
        tensor = torch.rand(3, 256, 256)
        to_pil = transforms.ToPILImage()
        img = to_pil(tensor)
        correct_pipeline = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        custom_pipeline = transforms.Compose([RandomResizedFlipCrop(224), transforms.ToTensor()])
        RandomResizedFlipCrop.get_params = self.deterministic_get_params
        transforms.RandomResizedCrop.get_params = self.deterministic_get_params
        torch.manual_seed(0)
        custom = custom_pipeline(img)
        torch.manual_seed(0)
        correct = correct_pipeline(img)
        assert torch.allclose(custom, correct, atol=1e-06)


    @pytest.mark.category1
    def test_RandomResizedFlipCrop_tensor(self):
        img = torch.randint(low=0, high=255, size=(3, 256, 256), dtype=torch.uint8)
        correct_pipeline = transforms.Compose([transforms.ToPILImage(),
                                               transforms.RandomResizedCrop(100),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        custom_pipeline = transforms.Compose([RandomResizedFlipCrop(100), NormalizeToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        RandomResizedFlipCrop.get_params = self.deterministic_get_params
        transforms.RandomResizedCrop.get_params = self.deterministic_get_params
        torch.manual_seed(0)
        custom = custom_pipeline(img)
        torch.manual_seed(0)
        correct = correct_pipeline(img)
        assert torch.allclose(custom, correct, atol=1e-06)


    @pytest.mark.category1
    def test_ipu_side_normalization(self):
        img = torch.rand(3, 100, 100)
        host_pipeline = NormalizeToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        model = models.NormalizeInputModel(lambda x: x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        host_result = host_pipeline(img)
        ipu_result = model(img)
        assert torch.allclose(host_result, ipu_result, atol=1e-06)


class TestHostBenchmark:
    @pytest.mark.category2
    def test_host_benchmark_cifar10(self):
        output = run_script("host_benchmark.py", "--data cifar10 --batch-size 256")
        assert "Throughput of the epoch" in output

    @pytest.mark.category2
    def test_poprun_host_benchmark(self):
        output = run_script("poprun", "--num-instances=2 --numa-aware=yes --offline-mode=1 --num-replicas=2 python host_benchmark.py --data cifar10 --batch-size 256", python=False)
        assert "Throughput of the epoch" in output
