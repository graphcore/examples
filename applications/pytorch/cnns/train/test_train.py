# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import gc
import subprocess
import pytest
import shutil
import torch
import poptorch
import popart
from poptorch.optim import SGD
from train import TrainingModelWithLoss
import sys
sys.path.append('..')
import datasets
import models


def run_script(script_name, parameters, python=True):
    cwd = os.path.dirname(os.path.abspath(__file__))
    param_list = parameters.split(" ")
    if python:
        cmd = ["python", script_name]
    else:
        cmd = [script_name]
    cmd = cmd + param_list
    out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.PIPE).decode("utf-8")
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


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_recomputation_checkpoints():
    gc.collect()
    # run the model with and without recomputation

    def train(model, recompute):
        input_data = torch.ones(1, 3, 224, 224)
        labels_data = torch.ones(1).long()
        model_opts = poptorch.Options()
        if recompute:
            model_opts._Popart.set("autoRecomputation", int(popart.RecomputationType.Standard))
        model_opts.anchorMode(poptorch.AnchorMode.All)
        model_opts.randomSeed(0)
        model_opts.Training.gradientAccumulation(1)
        model_opts.Precision.enableStochasticRounding(False)
        model_with_loss = TrainingModelWithLoss(model)
        optimizer = SGD(model_with_loss.parameters(), lr=0.01, momentum=0., use_combined_accum=True)
        training_model = poptorch.trainingModel(model_with_loss, model_opts, optimizer=optimizer)
        predictions = []
        for _ in range(3):
            preds, loss, _ = training_model(input_data, labels_data)
            predictions.append(preds)
        training_model.destroy()
        return predictions

    class Options():
        def __init__(self):
            self.model = "resnet18"
            self.precision = "16.16"
            self.norm_type = "group"
            self.norm_num_groups = 32
            self.full_precision_norm = False
            self.normalization_location = "none"
            self.pipeline_splits = []
            self.eight_bit_io = False
    opts = Options()
    torch.manual_seed(0)
    model = models.get_model(opts, datasets.datasets_info["cifar10"], pretrained=True)
    no_recompute_predictions = train(model, False)
    opts.recompute_checkpoints = ["conv", "norm"]
    torch.manual_seed(0)
    model = models.get_model(opts, datasets.datasets_info["cifar10"], pretrained=True)
    recompute_predictions = train(model, True)
    for pred1, pred2 in zip(no_recompute_predictions, recompute_predictions):
        assert torch.allclose(pred1, pred2, atol=1e-04)


@pytest.mark.category1
@pytest.mark.ipus(4)
def test_replicas_reduction():
    gc.collect()

    def common_model_opts():
        model_opts = poptorch.Options()
        model_opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)
        model_opts.anchorMode(poptorch.AnchorMode.All)
        model_opts.randomSeed(0)
        model_opts.Training.gradientAccumulation(1)
        return model_opts

    def run_model(model_opts):
        input_data = torch.ones(4, 1)
        labels_data = torch.ones(4).long()
        model = torch.nn.Linear(1, 2, bias=False)
        model_with_loss = TrainingModelWithLoss(model, 0.1)
        optimizer = SGD(model_with_loss.parameters(), lr=0.1, momentum=0., use_combined_accum=True)
        training_model = poptorch.trainingModel(model_with_loss, model_opts, optimizer=optimizer)
        for _ in range(3):
            preds, loss, _ = training_model(input_data, labels_data)
        # return the weights of the model
        return list(model_with_loss.model.named_parameters())[0][1], loss

    # Single replica
    model_opts = common_model_opts()
    model_opts.replicationFactor(1)
    single_replica_weights, single_replica_loss = run_model(model_opts)
    # 4 replica running
    gc.collect()
    model_opts = common_model_opts()
    model_opts.replicationFactor(4)
    replicated_weights, replicated_loss = run_model(model_opts)

    assert torch.allclose(single_replica_weights, replicated_weights, atol=1e-05)
    assert torch.allclose(single_replica_loss, replicated_loss, atol=1e-05)


@pytest.mark.category3
@pytest.mark.ipus(1)
def test_generated():
    gc.collect()
    run_script("train.py", f"--data generated --model resnet18 --epoch 1 --precision 16.16 --validation-mode none --optimizer sgd_combined --lr 0.001 --gradient-accumulation 128 --batch-size 1 --dataloader-worker 4 --seed 0")


@pytest.mark.category3
@pytest.mark.ipus(1)
@pytest.mark.parametrize("precision", ["16.16", "32.32"])
def test_synthetic(precision):
    gc.collect()
    run_script("train.py", f"--data synthetic --model resnet18 --epoch 1 --precision {precision} --validation-mode none --optimizer sgd_combined --lr 0.001 --gradient-accumulation 64 --batch-size 1 --dataloader-worker 4 --seed 0")


@pytest.mark.parametrize("label_smoothing", [0.0, 1.0, 0.1, 0.5])
def test_loss_function(label_smoothing):
    torch.manual_seed(0)
    inp = torch.rand(4, 10) * 10 - 5  # create random input between [-5,5)
    label = torch.ones(4).long()
    # calculate the ground truth
    log_pred = torch.nn.functional.log_softmax(inp, dim=-1)
    ground_truth = - torch.mean(torch.sum((label_smoothing / 10.0) * log_pred, dim=1) + (1.0 - label_smoothing) * log_pred[:, 1])
    model_with_loss = TrainingModelWithLoss(lambda x: x, label_smoothing=label_smoothing)
    _, loss, _ = model_with_loss(inp, label)
    assert torch.allclose(ground_truth, loss, atol=1e-05)


@pytest.mark.category3
@pytest.mark.ipus(1)
def test_mixup():
    gc.collect()
    run_script("train.py", f"--mixup-alpha 0.1 --data generated --model resnet18 --epoch 1 --validation-mode none --optimizer sgd_combined --batch-size 3 --dataloader-worker 1 --seed 0")


class TestSynthetic:
    @pytest.mark.category3
    @pytest.mark.ipus(2)
    @pytest.mark.ipu_version("ipu2")
    def test_synthetic_mixed_precision(self):
        gc.collect()
        run_script("train.py", "--data synthetic --model resnet18 --epoch 1 --precision 16.32 --pipeline-splits layer4/0 "
                   "--validation-mode none --optimizer sgd_combined --lr 0.001 --gradient-accumulation 64 --dataloader-worker 4 --seed 0")


class TestTrainCIFAR10:
    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_single_ipu_validation_groupnorm(self):
        gc.collect()
        out = run_script("train.py", "--data cifar10 --model resnet18 --epoch 3 --precision 16.16 --optimizer sgd_combined --lr 0.1 --batch-size 2 --gradient-accumulation 32 "
                                     "--norm-type group --norm-num-groups 32 --enable-stochastic-rounding --dataloader-worker 4 --seed 0")
        acc = get_test_accuracy(out)
        assert acc > 15.0


    @pytest.mark.category3
    @pytest.mark.ipus(1)
    @pytest.mark.ipu_version("ipu2")
    def test_single_ipu_validation_batchnorm(self):
        gc.collect()
        out = run_script("train.py", "--data cifar10 --model resnet18 --epoch 2 --precision 16.16 --optimizer sgd_combined --lr 0.1 --gradient-accumulation 8 "
                                     "--norm-type batch --batch-size 16 --enable-stochastic-rounding --dataloader-worker 4 --seed 0")
        acc = get_test_accuracy(out)
        assert acc > 15.0


    @pytest.mark.category3
    @pytest.mark.ipus(2)
    def test_replicas(self):
        gc.collect()
        out = run_script("train.py", "--data cifar10 --model resnet18 --epoch 2 --replicas 2 --precision 16.16 --validation-mode none --optimizer sgd_combined --lr 0.1 "
                                     "--gradient-accumulation 32 --enable-stochastic-rounding --dataloader-worker 4 --seed 0")
        acc = get_train_accuracy(out)
        assert acc > 15.0


    @pytest.mark.skip(reason="T42276")
    @pytest.mark.category3
    @pytest.mark.ipus(2)
    def test_efficient_net(self):
        gc.collect()
        out = run_script("train.py", "--data cifar10 --epoch 2 --model efficientnet-b0 --precision 16.16 --validation-mode none --optimizer sgd_combined --lr 0.1 --gradient-accumulation 64 "
                                     "--pipeline-splits _blocks/4/_bn1 --norm-type group --norm-num-groups 4 --enable-stochastic-rounding --dataloader-worker 4 --seed 0")
        acc = get_train_accuracy(out)
        assert acc > 15.0


    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_full_precision(self):
        gc.collect()
        out = run_script("train.py", "--data cifar10 --epoch 2 --model resnet18 --precision 32.32 --optimizer sgd_combined --lr 0.1 --batch-size 1 --gradient-accumulation 64 --dataloader-worker 4 --seed 0")
        acc = get_train_accuracy(out)
        assert acc > 15.0


    @pytest.mark.category3
    @pytest.mark.ipus(2)
    @pytest.mark.ipu_version("ipu2")
    def test_mixed_precision(self):
        gc.collect()
        out = run_script("train.py", "--data cifar10 --epoch 2 --model resnet18 --pipeline-splits layer4/0 --precision 16.32 --optimizer sgd_combined "
                                     "--lr 0.1 --batch-size 1 --gradient-accumulation 64 --validation-mode none --dataloader-worker 4 --seed 0")
        acc = get_train_accuracy(out)
        assert acc > 15.0


class TestRestoreCheckpoint:
    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_restore_train(self):
        gc.collect()
        # create a model
        out = run_script("train.py", "--data cifar10 --epoch 2 --model resnet18 --precision 16.16 --optimizer sgd_combined --lr 0.1 --batch-size 2 --gradient-accumulation 32 --seed 0 "
                                     "--validation-mode none --norm-type group --norm-num-groups 32 --checkpoint-path restore_test_path_test_restore_train --dataloader-worker 4")
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
        gc.collect()
        # create a model
        out = run_script("train.py", "--data cifar10 --epoch 1 --model resnet18 --precision 16.16 --optimizer sgd_combined --lr 0.1 --batch-size 2 --gradient-accumulation 32 --seed 0 "
                                     "--norm-type group --norm-num-groups 32 --checkpoint-path restore_test_path_test_validation --dataloader-worker 4")
        saved_test_acc = get_test_accuracy(out)
        # validate the model
        out = run_script("validate.py", "--checkpoint-path restore_test_path_test_validation/resnet18_cifar10_1.pt")
        acc = get_test_accuracy(out)
        # close enough
        assert abs(saved_test_acc - acc) < 0.01
        # remove folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        shutil.rmtree(os.path.join(script_dir, "restore_test_path_test_validation"))


    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_weight_avg(self):
        gc.collect()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out1 = run_script("train.py", "--data cifar10 --epoch 3 --model resnet18 --precision 16.16 --weight-avg-strategy mean --norm-type group "
                          "--norm-num-groups 32 --optimizer sgd_combined --lr 0.1 --batch-size 2 --gradient-accumulation 32 --checkpoint-path restore_test_path_weight_avg "
                          "--weight-avg-N 2 --dataloader-worker 4 --seed 0")
        os.remove(os.path.join(script_dir, "restore_test_path_weight_avg", "resnet18_cifar10_3_averaged.pt"))
        _ = run_script("weight_avg.py", "--checkpoint-path restore_test_path_weight_avg --weight-avg-strategy mean --weight-avg-N 2")
        out2 = run_script("validate.py", "--checkpoint-path restore_test_path_weight_avg/resnet18_cifar10_3_averaged.pt")
        acc1 = get_test_accuracy(out1)
        acc2 = get_test_accuracy(out1)
        assert acc1 > 15
        assert acc1 == acc2
        shutil.rmtree(os.path.join(script_dir, "restore_test_path_weight_avg"))

    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_mixup_validation_weight_avg(self):
        # Only make sure that checkpoint loading works with mixup model wrapper.
        gc.collect()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        run_script("train.py", f"--mixup-alpha 0.1 --data generated --checkpoint-path test_mixup_validation_weight_avg --weight-avg-strategy exponential --weight-avg-exp-decay 0.97 --model resnet18 --epoch 2 --validation-mode after --optimizer sgd_combined --batch-size 3 --dataloader-worker 1 --seed 0")
        shutil.rmtree(os.path.join(script_dir, "test_mixup_validation_weight_avg"))

    @pytest.mark.category3
    @pytest.mark.ipus(1)
    def test_mixup_restore_train(self):
        # Only make sure that checkpoint loading works with mixup model wrapper.
        gc.collect()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        run_script("train.py", f"--mixup-alpha 0.1 --data generated --checkpoint-path test_mixup_restore_train --model resnet18 --epoch 2 --validation-mode none --optimizer sgd_combined --batch-size 3 --dataloader-worker 1 --seed 0")
        run_script("restore.py", "--checkpoint-path test_mixup_restore_train/resnet18_generated_1.pt")
        shutil.rmtree(os.path.join(script_dir, "test_mixup_restore_train"))


class TestPopDist:
    @pytest.mark.category3
    @pytest.mark.ipus(4)
    def test_popdist(self):
        gc.collect()
        out = run_script("poprun", "--mpi-global-args='--allow-run-as-root' --num-instances=2 --numa-aware=yes --num-replicas=4 --ipus-per-replica 1 python train.py --data cifar10 --model resnet18 --epoch 2 "
                                   "--precision 16.16 --optimizer sgd_combined --lr 0.1 --batch-size 2 --gradient-accumulation 8 --enable-stochastic-rounding --validation-mode after --dataloader-worker 4 "
                                   "--norm-type group --norm-num-groups 32", python=False)
        train_acc = get_train_accuracy(out)
        assert train_acc > 15.0
        test_acc = get_test_accuracy(out)
        assert test_acc > 15.0
