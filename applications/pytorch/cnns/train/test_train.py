# Copyright 2020 Graphcore Ltd.
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
import models  # noqa: E402


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


class TestWeightSync(unittest.TestCase):
    @staticmethod
    def create_opts():
        class Opts:
            def __init__(self):
                self.model = "resnet18"
                self.pipeline_splits = ""
                self.precision = "full"
                self.normlayer = "batch"
                self.data = "imagenet"
        return Opts()

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_implicit_weightcopy_inference(self):
        x = torch.ones([1] + list(datasets_info["cifar10"]["in"]))
        opts = TestWeightSync.create_opts()
        model = models.get_model(opts, datasets_info["imagenet"], pretrained=True)
        model.eval()
        poptorch_model = poptorch.inferenceModel(model)
        y = model(x)
        for _ in range(10):
            y_poptorch = poptorch_model(x)
            assert torch.allclose(y, y_poptorch, atol=0.0001)

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_explicit_weightcopy_inference(self):
        x = torch.ones([1] + list(datasets_info["cifar10"]["in"]))
        opts = TestWeightSync.create_opts()
        model = models.get_model(opts, datasets_info["imagenet"], pretrained=True)
        model.eval()
        poptorch_model = poptorch.inferenceModel(model)
        y = model(x)
        for _ in range(10):
            poptorch_model.copyWeightsToDevice()
            y_poptorch = poptorch_model(x)
            assert torch.allclose(y, y_poptorch, atol=0.0001)


class TestTrainCIFAR10(unittest.TestCase):
    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_single_ipu(self):
        out = run_script("train.py", "--data cifar10 --model resnet18 --epoch 4 --precision full --no-validation --lr 0.1 --gradient-accumulation 64")
        acc = get_train_accuracy(out)
        self.assertGreater(acc, 20.0)


    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_single_ipu_validation_groupnorm(self):
        out = run_script("train.py", "--data cifar10 --model resnet18 --epoch 10 --precision full --lr 0.1 --gradient-accumulation 64 --normlayer group")
        acc = get_test_accuracy(out)
        self.assertGreater(acc, 20.0)


    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_single_ipu_validation_batchnorm(self):
        out = run_script("train.py", "--data cifar10 --model resnet18 --epoch 10 --precision half --lr 0.1 --gradient-accumulation 32 --normlayer batch --batch-size 2")
        acc = get_test_accuracy(out)
        self.assertGreater(acc, 20.0)


    @pytest.mark.category3
    @pytest.mark.ipus(2)
    def test_replicas(self):
        out = run_script("train.py", "--data cifar10 --model resnet18 --epoch 4 --replicas 2 --precision full --no-validation --lr 0.1 --gradient-accumulation 32")
        acc = get_train_accuracy(out)
        self.assertGreater(acc, 20.0)


    @pytest.mark.category3
    @pytest.mark.ipus(2)
    def test_efficient_net(self):
        out = run_script("train.py", "--data cifar10 --epoch 4 --model efficientnet-b0 --precision half --no-validation --lr 0.1 --gradient-accumulation 64 --pipeline-splits _blocks/4/_bn1 --normlayer group --groupnorm-group-num 4")
        acc = get_train_accuracy(out)
        self.assertGreater(acc, 20.0)


    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_halfprecision(self):
        out = run_script("train.py", "--data cifar10 --epoch 4 --model resnet18 --precision half --lr 0.1 --batch-size 1 --gradient-accumulation 64")
        acc = get_train_accuracy(out)
        self.assertGreater(acc, 20.0)


    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_group_normalization(self):
        out = run_script("train.py", "--data cifar10 --epoch 4 --model resnet18 --precision full --normlayer group --no-validation --lr 0.1 --gradient-accumulation 64")
        acc = get_train_accuracy(out)
        self.assertGreater(acc, 20.0)


class TestRestoreCheckpoint(unittest.TestCase):
    def setup_method(self, method):
        method = method.__name__

        script_dir = os.path.dirname(os.path.abspath(__file__))
        out = run_script("train.py", "--data cifar10 --epoch 2 " +
                         "--model resnet18 --data cifar10 " +
                         "--checkpoint-path restore_test_path_" + method + " "
                         "--precision full --lr 0.1 " +
                         "--gradient-accumulation 64 --normlayer group")
        self.saved_train_acc = get_train_accuracy(out)
        self.saved_test_acc = get_test_accuracy(out)
        # rename last checkpoint for the validation: this prevents an overwrite of the restoring
        os.rename(
            os.path.join(script_dir, "restore_test_path_" + method,
                         "resnet18_cifar10_2.pt"),
            os.path.join(script_dir, "restore_test_path_" + method,
                         "resnet18_validation.pt"))

    def teardown_method(self, method):
        method = method.__name__

        script_dir = os.path.dirname(os.path.abspath(__file__))
        shutil.rmtree(os.path.join(script_dir, "restore_test_path_" + method))

    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_restore_train(self):
        out = run_script("restore.py", "--checkpoint-path restore_test_path_test_restore_train/resnet18_cifar10_1.pt")
        acc = get_train_accuracy(out)
        self.assertGreater(acc, self.saved_train_acc-5.0)


    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_validation(self):
        out = run_script("validate.py", "--checkpoint-path restore_test_path_test_validation/resnet18_validation.pt")
        acc = get_test_accuracy(out)
        self.assertEqual(self.saved_test_acc, acc)


class TestGroupNormConversion(unittest.TestCase):
    @staticmethod
    def create_opts():
        class Opts:
            def __init__(self):
                self.normlayer = "group"
                self.groupnorm_group_num = 2
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
