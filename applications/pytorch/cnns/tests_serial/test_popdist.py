# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import gc
import os
import shutil
import import_helper
from utils import get_train_accuracy, get_test_accuracy, run_script, get_max_thoughput


class TestPopDist:
    @pytest.mark.category3
    @pytest.mark.ipus(2)
    def test_popdist_train(self):
        gc.collect()
        out = run_script("poprun", "--mpi-global-args='--allow-run-as-root' --num-instances=2 --numa-aware=yes --num-replicas=2 --ipus-per-replica 1 python train/train.py --data cifar10 --model resnet18 --epoch 2 "
                                   "--precision 16.16 --optimizer sgd_combined --lr 0.1 --batch-size 2 --gradient-accumulation 16 --enable-stochastic-rounding --validation-mode after --dataloader-worker 4 "
                                   "--norm-type group --norm-num-groups 32 --checkpoint-path restore_test_path_test_validation_distributed", python=False)
        train_acc = get_train_accuracy(out)
        assert train_acc > 15.0, "training accuracy not improved"
        test_acc = get_test_accuracy(out)
        assert test_acc > 15.0, "validation accuracy not improved"
        # Check the validation accuracy from a single instance
        out = run_script("train/validate.py", "--checkpoint-path restore_test_path_test_validation_distributed/resnet18_cifar10_2.pt")
        restored_test_acc = get_test_accuracy(out)
        assert abs(restored_test_acc - test_acc) < 0.01, "distributed and single  instance validation accuracies doesn't match"
        # remove folder
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        shutil.rmtree(os.path.join(parent_dir, "restore_test_path_test_validation_distributed"))


    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_popdist_inference(self):
        out = run_script("poprun", f"--mpi-global-args='--allow-run-as-root' --num-instances=2 --numa-aware=yes --num-replicas=2 python inference/run_benchmark.py --data generated --model resnet18 --batch-size 4 --precision 16.16 --iterations 10 --dataloader-worker 4", python=False)
        max_thoughput = get_max_thoughput(out)
        assert max_thoughput > 0
