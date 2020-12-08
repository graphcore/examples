# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import subprocess
import unittest
import pytest
import shutil
from data import datasets_info
import poptorch
import threading
import torch
import sys
sys.path.append('..')
import models


def run_script(script_name, parameters):
    cwd = os.path.dirname(os.path.abspath(__file__))
    param_list = parameters.split(" ")
    cmd = ["python3", script_name] + param_list
    out = subprocess.check_output(cmd, cwd=cwd).decode("utf-8")
    return out


def get_test_accuracy(output):
    prefix = "Accuracy on test set:"
    pos_start = output.rfind(prefix)
    pos_end = pos_start + output[pos_start:].find("%")
    return float(output[pos_start+len(prefix)+1:pos_end])


def get_train_accuracy(output):
    prefix = "Train accuracy is"
    pos_start = output.rfind(prefix)
    pos_end = pos_start + output[pos_start:].find("%")
    return float(output[pos_start+len(prefix)+1:pos_end])


@pytest.mark.category3
@pytest.mark.ipus(1)
@pytest.mark.parametrize("precision", ["16.16", "32.32"])
def test_synthetic(precision):
    run_script("train.py", f"--data synthetic --model resnet18 --epoch 1 --precision {precision} --no-validation --lr 0.001 --gradient-accumulation 64 --batch-size 1")


class TestSynthetic:
    @pytest.mark.category3
    @pytest.mark.ipus(2)
    def test_synthetic_mixed_precision(self):
        run_script("train.py", "--data synthetic --model resnet18 --epoch 1 --precision 16.32 --pipeline-splits layer4/0 --enable-pipeline-recompute "
                   "--no-validation --lr 0.001 --gradient-accumulation 64")


class TestTrainCIFAR10:
    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_single_ipu_validation_groupnorm(self):
        out = run_script("train.py", "--data cifar10 --model resnet18 --epoch 3 --precision 16.16 --lr 0.1 --batch-size 2 --gradient-accumulation 32 "
                                     "--norm-type group --norm-num-groups 32 --enable-stochastic-rounding")
        acc = get_test_accuracy(out)
        assert acc > 15.0


    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_single_ipu_validation_batchnorm(self):
        out = run_script("train.py", "--data cifar10 --model resnet18 --epoch 3 --precision 16.16 --lr 0.1 --gradient-accumulation 32 "
                                     "--norm-type batch --batch-size 2 --enable-stochastic-rounding")
        acc = get_test_accuracy(out)
        assert acc > 15.0


    @pytest.mark.category3
    @pytest.mark.ipus(2)
    def test_replicas(self):
        out = run_script("train.py", "--data cifar10 --model resnet18 --epoch 2 --replicas 2 --precision 16.16 --no-validation --lr 0.1 "
                                     "--gradient-accumulation 32 --enable-stochastic-rounding")
        acc = get_train_accuracy(out)
        assert acc > 15.0


    @pytest.mark.category3
    @pytest.mark.ipus(2)
    def test_efficient_net(self):
        out = run_script("train.py", "--data cifar10 --epoch 2 --model efficientnet-b0 --precision 16.16 --no-validation --lr 0.1 --gradient-accumulation 64 "
                                     "--pipeline-splits _blocks/4/_bn1 --norm-type group --norm-num-groups 4 --enable-stochastic-rounding")
        acc = get_train_accuracy(out)
        assert acc > 15.0


    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_full_precision(self):
        out = run_script("train.py", "--data cifar10 --epoch 2 --model resnet18 --precision 32.32 --lr 0.1 --batch-size 1 --gradient-accumulation 64")
        acc = get_train_accuracy(out)
        assert acc > 15.0


    @pytest.mark.category3
    @pytest.mark.ipus(2)
    def test_mixed_precision(self):
        out = run_script("train.py", "--data cifar10 --epoch 2 --model resnet18 --pipeline-splits layer4/0 --enable-pipeline-recompute --precision 16.32 "
                                     "--lr 0.1 --batch-size 1 --gradient-accumulation 64 --no-validation")
        acc = get_train_accuracy(out)
        assert acc > 15.0


class TestRestoreCheckpoint:
    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_restore_train(self):
        # create a model
        out = run_script("train.py", "--data cifar10 --epoch 2 --model resnet18 --precision 16.16 --lr 0.1 --batch-size 2 --gradient-accumulation 32 "
                                     "--no-validation --norm-type group --norm-num-groups 32 --checkpoint-path restore_test_path_test_restore_train")
        saved_train_acc = get_train_accuracy(out)
        # reload the model
        out = run_script("restore.py", "--checkpoint-path restore_test_path_test_restore_train/resnet18_cifar10_1.pt")
        acc = get_train_accuracy(out)
        assert acc > saved_train_acc - 5.0
        # remove folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        shutil.rmtree(os.path.join(script_dir, "restore_test_path_test_restore_train"))


    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_validation(self):
        # create a model
        out = run_script("train.py", "--data cifar10 --epoch 1 --model resnet18 --precision 16.16 --lr 0.1 --batch-size 2 --gradient-accumulation 32 "
                                     "--norm-type group --norm-num-groups 32 --checkpoint-path restore_test_path_test_validation")
        saved_test_acc = get_test_accuracy(out)
        # validate the model
        out = run_script("validate.py", "--checkpoint-path restore_test_path_test_validation/resnet18_cifar10_1.pt")
        acc = get_test_accuracy(out)
        # close enough
        assert abs(saved_test_acc - acc) < 0.01
        # remove folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        shutil.rmtree(os.path.join(script_dir, "restore_test_path_test_validation"))


class TestGroupNormConversion:
    @staticmethod
    def create_opts():
        class Opts:
            def __init__(self):
                self.norm_type = "group"
                self.norm_num_groups = 2
        return Opts()


    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_single_element_model(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.bn = torch.nn.BatchNorm2d(1)

        model = Model()
        opts = TestGroupNormConversion.create_opts()
        models.replace_bn(model, opts)

        assert isinstance(model.bn, torch.nn.GroupNorm)

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_sequential_model(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 2, 3),
                    torch.nn.BatchNorm2d(1)
                )

        model = Model()
        opts = TestGroupNormConversion.create_opts()
        models.replace_bn(model, opts)

        assert isinstance(model.layers[1], torch.nn.GroupNorm) and isinstance(model.layers[0], torch.nn.Conv2d)

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_nested_model(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super(Block, self).__init__()
                self.conv = torch.nn.Conv2d(1, 2, 3)
                self.bn = torch.nn.BatchNorm2d(1)

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.layers = torch.nn.Sequential(
                    Block(),
                    torch.nn.Sequential(
                        Block(),
                        torch.nn.BatchNorm2d(1)
                    ),
                    torch.nn.BatchNorm2d(1)
                )

        model = Model()
        opts = TestGroupNormConversion.create_opts()
        models.replace_bn(model, opts)

        assert isinstance(model.layers[0].conv, torch.nn.Conv2d) and \
            isinstance(model.layers[0].bn, torch.nn.GroupNorm) and \
            isinstance(model.layers[1][0].conv, torch.nn.Conv2d) and \
            isinstance(model.layers[1][0].bn, torch.nn.GroupNorm) and \
            isinstance(model.layers[2], torch.nn.GroupNorm)
