# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import gc
import pytest
import shutil
import torch
import poptorch
import popart
from poptorch.optim import SGD
import import_helper
from models.loss import TrainingModelWithLoss, LabelSmoothing
import datasets
import models
from utils import get_train_accuracy, get_test_accuracy, run_script


@pytest.mark.ipus(1)
def test_recomputation_checkpoints():
    gc.collect()
    # run the model with and without recomputation

    def train(model, recompute, args=None):
        input_data = torch.ones(1, 3, 224, 224)
        labels_data = torch.ones(1).long()
        if args and args.precision == "16.16":
            input_data = input_data.half()
        opts = poptorch.Options()
        if recompute:
            opts._Popart.set("autoRecomputation", int(popart.RecomputationType.Standard))
        opts.outputMode(poptorch.OutputMode.All)
        opts.randomSeed(0)
        opts.Training.gradientAccumulation(1)
        opts.Precision.enableStochasticRounding(False)
        model_with_loss = TrainingModelWithLoss(model, LabelSmoothing(label_smoothing=0.0).get_losses_list())
        optimizer = SGD(model_with_loss.parameters(), lr=0.01, momentum=0.0, use_combined_accum=True)
        training_model = poptorch.trainingModel(model_with_loss, opts, optimizer=optimizer)
        predictions = []
        for _ in range(3):
            preds, _, _ = training_model(input_data, labels_data)
            predictions.append(preds)
        training_model.destroy()
        return predictions

    class Options:
        def __init__(self):
            self.model = "resnet18"
            self.precision = "16.16"
            self.norm_type = "group"
            self.norm_eps = 1e-5
            self.norm_num_groups = 32
            self.normalization_location = "none"
            self.pipeline_splits = []
            self.eight_bit_io = False
            self.num_io_tiles = 0

    args = Options()
    torch.manual_seed(0)
    model = models.get_model(args, datasets.datasets_info["cifar10"], pretrained=False)
    no_recompute_predictions = train(model, False, args)
    args.recompute_checkpoints = ["conv", "norm"]
    torch.manual_seed(0)
    model = models.get_model(args, datasets.datasets_info["cifar10"], pretrained=False)
    recompute_predictions = train(model, True, args)
    for pred1, pred2 in zip(no_recompute_predictions, recompute_predictions):
        assert torch.allclose(pred1, pred2, atol=1e-04)


@pytest.mark.ipus(4)
def test_replicas_reduction():
    gc.collect()

    def common_opts():
        opts = poptorch.Options()
        opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)
        opts.outputMode(poptorch.OutputMode.All)
        opts.randomSeed(0)
        opts.Training.gradientAccumulation(1)
        return opts

    def run_model(opts):
        input_data = torch.ones(4, 1)
        labels_data = torch.ones(4).long()
        model = torch.nn.Linear(1, 2, bias=False)
        model_with_loss = TrainingModelWithLoss(model, LabelSmoothing(label_smoothing=0.0).get_losses_list())
        optimizer = SGD(model_with_loss.parameters(), lr=0.1, momentum=0.0, use_combined_accum=True)
        training_model = poptorch.trainingModel(model_with_loss, opts, optimizer=optimizer)
        for _ in range(3):
            loss, _, _ = training_model(input_data, labels_data)
        # return the weights of the model
        return (
            torch.broadcast_to(
                list(model_with_loss.model.named_parameters())[0][1],
                list(model_with_loss.model.named_parameters())[0][1].shape,
            ),
            loss,
        )

    # Single replica
    opts = common_opts()
    opts.replicationFactor(1)
    single_replica_weights, single_replica_loss = run_model(opts)
    # 4 replica running
    gc.collect()
    opts = common_opts()
    opts.replicationFactor(4)
    replicated_weights, replicated_loss = run_model(opts)

    assert torch.allclose(single_replica_weights, replicated_weights, atol=1e-05)
    assert torch.allclose(single_replica_loss, replicated_loss, atol=1e-05)


@pytest.mark.ipus(1)
@pytest.mark.parametrize("precision", ["16.16", "32.32"])
def test_synthetic(precision):
    gc.collect()
    run_script(
        "train/train.py",
        f"--data synthetic --model resnet18 --epoch 1 --precision {precision} --validation-mode none --optimizer sgd_combined --lr 0.001 --gradient-accumulation 64 --micro-batch-size 1 --dataloader-worker 4 --seed 0",
    )


@pytest.mark.parametrize("label_smoothing", [0.0, 1.0, 0.1, 0.5])
def test_loss_function(label_smoothing):
    torch.manual_seed(0)
    inp = torch.rand(4, 10) * 10 - 5  # create random input between [-5,5)
    label = torch.ones(4).long()
    # calculate the ground truth
    log_pred = torch.nn.functional.log_softmax(inp, dim=-1)
    ground_truth = -torch.mean(
        torch.sum((label_smoothing / 10.0) * log_pred, dim=1) + (1.0 - label_smoothing) * log_pred[:, 1]
    )
    model_with_loss = TrainingModelWithLoss(
        lambda x: x, LabelSmoothing(label_smoothing=label_smoothing).get_losses_list()
    )
    loss, _, _ = model_with_loss(inp, label)
    assert torch.allclose(ground_truth, loss, atol=1e-05)


# class TestSynthetic:
#     @pytest.mark.ipus(2)
#     @pytest.mark.ipu_version("ipu2")
#     def test_synthetic_mixed_precision(self):
#         gc.collect()
#         run_script("train/train.py", "--data synthetic --model resnet18 --epoch 1 --precision 16.32 --pipeline-splits layer4/0 "
#                    "--validation-mode none --optimizer sgd_combined --lr 0.001 --gradient-accumulation 64 --dataloader-worker 4 --seed 0")


class TestTrainCIFAR10:
    @pytest.mark.ipus(1)
    def test_single_ipu_validation_groupnorm(self):
        gc.collect()
        out = run_script(
            "train/train.py",
            "--data cifar10 --model resnet18 --epoch 3 --precision 16.16 --optimizer sgd_combined --lr 0.1 --micro-batch-size 2 --gradient-accumulation 32 "
            "--norm-type group --norm-num-groups 32 --enable-stochastic-rounding --dataloader-worker 4 --seed 0",
        )
        acc = get_test_accuracy(out)
        assert acc > 15.0

    @pytest.mark.ipus(1)
    @pytest.mark.ipu_version("ipu2")
    def test_single_ipu_validation_batchnorm(self):
        gc.collect()
        out = run_script(
            "train/train.py",
            "--data cifar10 --model resnet18 --epoch 2 --precision 16.16 --optimizer sgd_combined --lr 0.1 --gradient-accumulation 8 "
            "--norm-type batch --micro-batch-size 16 --enable-stochastic-rounding --dataloader-worker 4 --seed 0",
        )
        acc = get_test_accuracy(out)
        assert acc > 15.0

    @pytest.mark.ipus(2)
    def test_replicas(self):
        gc.collect()
        out = run_script(
            "train/train.py",
            "--data cifar10 --model resnet18 --epoch 2 --replicas 2 --precision 16.16 --validation-mode none --optimizer sgd_combined --lr 0.1 "
            "--gradient-accumulation 32 --enable-stochastic-rounding --dataloader-worker 4 --seed 0",
        )
        acc = get_train_accuracy(out)
        assert acc > 15.0

    @pytest.mark.ipus(1)
    def test_full_precision(self):
        gc.collect()
        out = run_script(
            "train/train.py",
            "--data cifar10 --model resnet18 --epoch 2 --replicas 1 --precision 32.32 --validation-mode none --optimizer sgd_combined --lr 0.1 "
            "--gradient-accumulation 64 --micro-batch-size 1 --dataloader-worker 4 --seed 0",
        )
        acc = get_train_accuracy(out)
        assert acc > 15.0

    # @pytest.mark.ipus(2)
    # @pytest.mark.ipu_version("ipu2")
    # def test_mixed_precision(self):
    #     gc.collect()
    #     out = run_script("train/train.py", "--data cifar10 --epoch 2 --model resnet18 --pipeline-splits layer4/0 --precision 16.32 --optimizer sgd_combined "
    #                                        "--lr 0.1 --micro-batch-size 1 --gradient-accumulation 64 --validation-mode none --dataloader-worker 4 --seed 0")
    #     acc = get_train_accuracy(out)
    #     assert acc > 15.0

    # @pytest.mark.ipus(1)
    # @pytest.mark.ipu_version("ipu2")
    # def test_half_resolution_training(self):
    #     gc.collect()
    #     out = run_script("train/train.py", "--data cifar10 --model resnet18 --epoch 1 --precision 16.32 --optimizer sgd_combined --lr 0.1 --micro-batch-size 2 --gradient-accumulation 32 "
    #                                        "--norm-type batch --dataloader-worker 4 --half-res-training --fine-tune-epoch 1 --fine-tune-first-trainable-layer layer3 --weight-avg-strategy exponential "
    #                                        "--weight-avg-exp-decay 0.97 --checkpoint-output-dir test_half_resolution_training --checkpoint-input-dir test_half_resolution_training --seed 0")
    #     acc = get_test_accuracy(out)
    #     assert acc > 15.0
    #     # remove folder
    #     parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #     shutil.rmtree(os.path.join(parent_dir, "test_half_resolution_training"))


class TestRestoreCheckpoint:
    @pytest.mark.ipus(1)
    def test_restore_train(self):
        gc.collect()
        # create a model
        out = run_script(
            "train/train.py",
            "--data cifar10 --epoch 2 --model resnet18 --precision 16.16 --optimizer sgd_combined --lr 0.1 --micro-batch-size 2 --gradient-accumulation 32 --seed 0 "
            "--validation-mode none --norm-type group --norm-num-groups 32 --checkpoint-output-dir restore_test_path_test_restore_train --dataloader-worker 4",
        )
        saved_train_acc = get_train_accuracy(out)
        # reload the model
        out = run_script(
            "train/restore.py", "--checkpoint-input-path restore_test_path_test_restore_train/resnet18_cifar10_1.pt"
        )
        acc = get_train_accuracy(out)
        assert acc > saved_train_acc - 5.0
        # remove folder
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        shutil.rmtree(os.path.join(parent_dir, "restore_test_path_test_restore_train"))

    @pytest.mark.ipus(1)
    def test_validation(self):
        gc.collect()
        # create a model
        out = run_script(
            "train/train.py",
            "--data cifar10 --epoch 1 --model resnet18 --precision 16.16 --optimizer sgd_combined --lr 0.1 --micro-batch-size 2 --gradient-accumulation 32 --seed 0 "
            "--norm-type group --norm-num-groups 32 --checkpoint-output-dir restore_test_path_test_validation --dataloader-worker 4",
        )
        saved_test_acc = get_test_accuracy(out)
        # validate the model
        out = run_script(
            "train/validate.py", "--checkpoint-input-path restore_test_path_test_validation/resnet18_cifar10_1.pt"
        )
        acc = get_test_accuracy(out)
        # close enough
        assert abs(saved_test_acc - acc) < 0.01
        # remove folder
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        shutil.rmtree(os.path.join(parent_dir, "restore_test_path_test_validation"))

    @pytest.mark.ipus(1)
    @pytest.mark.ipu_version("ipu2")
    def test_weight_avg(self):
        gc.collect()
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out1 = run_script(
            "train/train.py",
            "--data cifar10 --epoch 3 --model resnet18 --precision 16.16 --weight-avg-strategy mean --norm-type group "
            "--norm-num-groups 32 --optimizer sgd_combined --lr 0.1 --micro-batch-size 2 --gradient-accumulation 32 --checkpoint-output-dir restore_test_path_weight_avg "
            "--checkpoint-input-dir restore_test_path_weight_avg --weight-avg-N 2 --dataloader-worker 4 --seed 0",
        )
        os.remove(os.path.join(parent_dir, "restore_test_path_weight_avg", "resnet18_cifar10_3_averaged.pt"))
        _ = run_script(
            "train/weight_avg.py",
            "--checkpoint-input-path restore_test_path_weight_avg --checkpoint-output-path restore_test_path_weight_avg --weight-avg-strategy mean --weight-avg-N 2",
        )
        out2 = run_script(
            "train/validate.py", "--checkpoint-input-path restore_test_path_weight_avg/resnet18_cifar10_3_averaged.pt"
        )
        acc1 = get_test_accuracy(out1)
        acc2 = get_test_accuracy(out2)
        assert acc1 > 15
        assert (
            acc1 - 5 < acc2
        )  # acc1 should be lower most of the times, but with only three epochs it could be slightly greater than acc2 sometimes
        shutil.rmtree(os.path.join(parent_dir, "restore_test_path_weight_avg"))

    @pytest.mark.ipus(1)
    @pytest.mark.ipu_version("ipu2")
    def test_mixup_cutmix_validation_weight_avg(self):
        # Only make sure that checkpoint loading works with mixup model wrapper.
        gc.collect()
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        run_script(
            "train/train.py",
            f"--mixup-alpha 0.1 --cutmix-lambda-low 0.2 --cutmix-lambda-high 0.8 --data generated --checkpoint-output-dir test_mixup_cutmix_validation_weight_avg --checkpoint-input-dir test_mixup_cutmix_validation_weight_avg --weight-avg-strategy exponential --weight-avg-exp-decay 0.97 --model resnet18 --epoch 2 --validation-mode after --optimizer sgd_combined --micro-batch-size 4 --dataloader-worker 1 --seed 0",
        )
        shutil.rmtree(os.path.join(parent_dir, "test_mixup_cutmix_validation_weight_avg"))

    @pytest.mark.ipus(1)
    @pytest.mark.ipu_version("ipu2")
    def test_mixup_cutmix_restore_train(self):
        # Only make sure that checkpoint loading works with mixup model wrapper.
        gc.collect()
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        run_script(
            "train/train.py",
            f"--mixup-alpha 0.1 --cutmix-lambda-low 0.5 --cutmix-lambda-high 0.5 --data generated --checkpoint-output-dir test_mixup_cutmix_restore_train --model resnet18 --epoch 2 --validation-mode none --optimizer sgd_combined --micro-batch-size 4 --dataloader-worker 1 --seed 0",
        )
        run_script(
            "train/restore.py", "--checkpoint-input-path test_mixup_cutmix_restore_train/resnet18_generated_1.pt"
        )
        shutil.rmtree(os.path.join(parent_dir, "test_mixup_cutmix_restore_train"))
