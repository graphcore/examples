# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import gc
import os
import shutil
import import_helper
from utils import (
    get_train_accuracy,
    get_test_accuracy,
    run_script,
    get_max_thoughput,
    get_current_interpreter_executable,
)


class TestPopDist:
    @pytest.mark.ipus(2)
    @pytest.mark.ipu_version("ipu2")
    def test_popdist_train(self):
        gc.collect()
        executable = get_current_interpreter_executable()
        out = run_script(
            "poprun",
            f"--num-instances=2 --num-replicas=2 {executable} train/train.py --data cifar10 --model resnet18 --epoch 2 "
            "--precision 16.16 --optimizer sgd_combined --lr 0.1 --micro-batch-size 2 --gradient-accumulation 16 --enable-stochastic-rounding --validation-mode after --dataloader-worker 4 "
            "--norm-type group --norm-num-groups 32 --checkpoint-output-dir restore_test_path_test_validation_distributed --checkpoint-input-dir restore_test_path_test_validation_distributed",
            python=False,
        )
        train_acc = get_train_accuracy(out)
        assert train_acc > 15.0, "training accuracy not improved"
        test_acc = get_test_accuracy(out)
        assert test_acc > 15.0, "validation accuracy not improved"
        # Check the validation accuracy from a single instance
        out = run_script(
            "train/validate.py",
            "--checkpoint-input-path restore_test_path_test_validation_distributed/resnet18_cifar10_2.pt",
        )
        restored_test_acc = get_test_accuracy(out)
        assert (
            abs(restored_test_acc - test_acc) < 0.01
        ), "distributed and single  instance validation accuracies doesn't match"
        # remove folder
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        shutil.rmtree(
            os.path.join(parent_dir, "restore_test_path_test_validation_distributed")
        )

    @pytest.mark.ipus(2)
    def test_popdist_inference(self):
        executable = get_current_interpreter_executable()
        out = run_script(
            "poprun",
            f"--num-instances=2 --num-replicas=2 {executable} inference/run_benchmark.py --data generated --model resnet18 --micro-batch-size 4 --precision 16.16 --iterations 10 --dataloader-worker 4",
            python=False,
        )
        max_thoughput = get_max_thoughput(out)
        assert max_thoughput > 0
