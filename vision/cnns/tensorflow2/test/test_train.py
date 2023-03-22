# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import sys
from pathlib import Path
import glob
import shutil
import pytest
from examples_tests.test_util import SubProcessChecker

sys.path.append(str(Path(__file__).absolute().parent.parent))
from test_common import run_train


class WrongArguments(SubProcessChecker):
    def test_unsupported_argument(self):
        with self.assertRaises(AssertionError):
            run_train(self, "--wrong-arg", "0")

    def test_wrong_device_assignment(self):
        with self.assertRaises(AssertionError):
            run_train(
                self,
                "--model-name",
                "cifar_resnet8",
                "--pipeline-splits",
                "conv2_block1_post_addition_relu",
                "conv3_block1_post_addition_relu",
                "conv4_block1_post_addition_relu",
                "--device-mapping",
                "0",
                "1",
                "2",
                "4",
                "--gradient-accumulation-count",
                "8",
                "--recomputation",
                "True",
            )

    def test_recomputation_no_pipeline(self):
        with self.assertRaises(AssertionError):
            run_train(
                self,
                "--weight-updates-per-epoch",
                "1",
                "--dataset",
                "imagenet",
                "--dataset-path",
                "/localdata/datasets/imagenet-data",
                "--accelerator-side-preprocess",
                "True",
                "--gradient-accumulation-count",
                "8",
                "--recomputation",
                "True",
            )

    def test_recompute_with_no_split(self):
        with self.assertRaises(AssertionError):
            run_train(self, "--recomputation", "True")

    def test_logs_per_epoch_neg(self):
        with self.assertRaises(AssertionError):
            run_train(self, "--logs-per-epoch", "-1")

    def test_logs_per_epoch_not_multiple_of_epoch(self):
        with self.assertRaises(AssertionError):
            run_train(self, "--logs-per-epoch", "0.25", "--num-epochs", "6")

    def test_non_accelerated_8_bit_io(self):
        with self.assertRaises(AssertionError) as e:
            run_train(
                self,
                "--accelerator-side-preprocess",
                "False",
                "--micro-batch-size",
                "8",
                "--validation",
                "--eight-bit-transfer",
                "True",
            )


class SimplePass(SubProcessChecker):
    def test_help(self):
        output = run_train(self, "--help")
        self.assertIn("usage", output)

    @pytest.mark.skip(reason="use of external data (T68092)")
    def test_one_update_mnist(self):
        path_to_mnist = "/localdata/datasets/mnist"
        if not os.path.exists(path_to_mnist):
            raise NameError(f"Directory {path_to_mnist} from TFDS should have been copied to CI for this test")
        output = run_train(
            self,
            "--weight-updates-per-epoch",
            "1",
            "--dataset",
            "mnist",
            "--label-smoothing",
            "0.1",
            "--dataset-path",
            "/localdata/datasets/",
        )
        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)

    @pytest.mark.skip(reason="use of external data (T68092)")
    def test_one_update_accelerator(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        output = run_train(
            self,
            "--weight-updates-per-epoch",
            "1",
            "--accelerator-side-preprocess",
            "True",
            "--eight-bit-transfer",
            "True",
            "--label-smoothing",
            "0.1",
            "--synthetic-data",
            "ipu",
            "--dataset-path",
            "/localdata/datasets/",
        )
        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)

    @pytest.mark.skip(reason="use of external data (T68092)")
    def test_first_ckpt_epoch(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        checkpoint_dir = "/tmp/first_ckpt_epoch_test/"
        output = run_train(
            self,
            "--weight-updates-per-epoch",
            "1",
            "--logs-per-epoch",
            "1",
            "--ckpts-per-epoch",
            "1/3",
            "--first-ckpt-epoch",
            "2",
            "--num-epochs",
            "5",
            "--checkpoint-output-dir",
            checkpoint_dir,
            "--clean-dir",
            "False",
        )
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.h5"))
        for idx, epoch in [(0, 2.0), (1, 5.0)]:
            self.assertTrue(checkpoint_files[idx].endswith(f"epoch_{epoch}.h5"))
        shutil.rmtree(checkpoint_dir)


@pytest.mark.skip(reason="use of external data (T68092)")
class Resnet8(SubProcessChecker):
    def test_config(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        output = run_train(self, "--weight-updates-per-epoch", "1", "--config", "resnet8_test")
        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)

    def test_logs_per_epoch_lt_one(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        output = run_train(
            self,
            "--model",
            "cifar_resnet8",
            "--dataset-path",
            "/localdata/datasets/",
            "--num-epochs",
            "3",
            "--precision",
            "16.16",
            "--micro-batch-size",
            "8",
            "--validation",
            "False",
            "--half-partials",
            "True",
            "--gradient-accumulation",
            "8",
            "--logs-per-epoch",
            "1/3",
        )
        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)

    @pytest.mark.skip(reason="use of external data (T68092)")
    def test_pipeline(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        output = run_train(
            self,
            "--model-name",
            "cifar_resnet8",
            "--dataset-path",
            "/localdata/datasets/",
            "--pipeline-splits",
            "conv2_block1_post_addition_relu",
            "conv3_block1_post_addition_relu",
            "conv4_block1_post_addition_relu",
            "--device-mapping",
            "0",
            "1",
            "2",
            "3",
            "--gradient-accumulation-count",
            "8",
            "--recomputation",
            "True",
        )
        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)


@pytest.mark.skip(reason="use of external data (T68092)")
class ImageNet(SubProcessChecker):
    def test_accelerated(self):
        path_to_imagenet = "/localdata/datasets/imagenet-data"
        if not os.path.exists(path_to_imagenet):
            raise NameError(f"Directory {path_to_imagenet} should have been copied to CI for this test")
        output = run_train(
            self,
            "--weight-updates-per-epoch",
            "1",
            "--dataset",
            "imagenet",
            "--dataset-path",
            "/localdata/datasets/imagenet-data",
            "--accelerator-side-preprocess",
            "True",
            "--micro-batch-size",
            "8",
            "--validation",
            "False",
            "--half-partials",
            "True",
            "--gradient-accumulation",
            "8",
            "--eight-bit-transfer",
            "True",
        )
        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)

    def test_non_accelerated(self):
        path_to_imagenet = "/localdata/datasets/imagenet-data"
        if not os.path.exists(path_to_imagenet):
            raise NameError(f"Directory {path_to_imagenet} should have been copied to CI for this test")
        output = run_train(
            self,
            "--weight-updates-per-epoch",
            "1",
            "--dataset",
            "imagenet",
            "--dataset-path",
            "/localdata/datasets/imagenet-data",
            "--accelerator-side-preprocess",
            "False",
            "--micro-batch-size",
            "8",
            "--validation",
            "False",
            "--half-partials",
            "True",
            "--precision",
            "16.16",
            "--gradient-accumulation",
            "8",
        )
        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)

    def test_fused_preprocessing(self):
        path_to_imagenet = "/localdata/datasets/imagenet-data"
        if not os.path.exists(path_to_imagenet):
            raise NameError(f"Directory {path_to_imagenet} should have been copied to CI for this test")
        output = run_train(
            self,
            "--weight-updates-per-epoch",
            "1",
            "--dataset",
            "imagenet",
            "--dataset-path",
            "/localdata/datasets/imagenet-data",
            "--accelerator-side-preprocess",
            "True",
            "--fused-preprocessing",
            "True",
            "--eight-bit-transfer",
            "True",
            "--micro-batch-size",
            "8",
            "--validation",
            "False",
            "--half-partials",
            "True",
            "--precision",
            "16.16",
        )
        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)

    def test_fused_preprocessing_on_host(self):
        path_to_imagenet = "/localdata/datasets/imagenet-data"
        if not os.path.exists(path_to_imagenet):
            raise NameError(f"Directory {path_to_imagenet} should have been copied to CI for this test")
        with self.assertRaises(AssertionError):
            output = run_train(
                self,
                "--dataset",
                "imagenet",
                "--dataset-path",
                "/localdata/datasets/imagenet-data",
                "--accelerator-side-preprocess",
                "False",
                "--fused-preprocessing",
                "True",
            )


@pytest.mark.skip(reason="use of external data (T68092)")
class DisableVariableOffloading(SubProcessChecker):
    def test_disable_variable_offloading_global_batch_size(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        output = run_train(
            self,
            "--weight-updates-per-epoch",
            "1",
            "--dataset-path",
            "/localdata/datasets/",
            "--global-batch-size",
            "10",
        )
        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)

    def test_both_args(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        with self.assertRaises(AssertionError):
            output = run_train(
                self,
                "--weight-updates-per-epoch",
                "1",
                "--dataset-path",
                "/localdata/datasets/",
                "--gradient-accumulation-count",
                "10",
                "--global-batch-size",
                "10",
            )


@pytest.mark.skip(reason="use of external data (T68092)")
class AvailableMemoryProportion(SubProcessChecker):
    def test_single_value_pipeline(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        output = run_train(
            self,
            "--weight-updates-per-epoch",
            "1",
            "--validation",
            "False",
            "--dataset-path",
            "/localdata/datasets/",
            "--available-memory-proportion",
            "50",
            "--pipeline-splits",
            "conv2d_1",
            "--gradient-accumulation-count",
            "4",
        )

        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)

    def test_multiple_values_no_pipelining(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        with self.assertRaises(AssertionError):
            output = run_train(
                self,
                "--weight-updates-per-epoch",
                "1",
                "--validation",
                "False",
                "--dataset-path",
                "/localdata/datasets/",
                "--available-memory-proportion",
                "50",
                "50",
            )

    def test_multiple_values_pipeline(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        output = run_train(
            self,
            "--weight-updates-per-epoch",
            "1",
            "--validation",
            "False",
            "--dataset-path",
            "/localdata/datasets/",
            "--available-memory-proportion",
            "50",
            "50",
            "60",
            "60",
            "--pipeline-splits",
            "conv2d_1",
            "--gradient-accumulation-count",
            "4",
        )

        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)

    def test_pipeline_mixed_precision_resnet50(self):
        path_to_imagenet = "/localdata/datasets/imagenet-data"
        if not os.path.exists(path_to_imagenet):
            raise NameError(f"Directory {path_to_imagenet} should have been copied to CI for this test")
        output = run_train(
            self,
            "--config",
            "resnet50_16ipus_16k_bn_pipeline",
            "--weight-updates-per-epoch",
            "1",
            "--num-epochs",
            "1",
            "--micro-batch-size",
            "1",
            "--label-smoothing",
            "0.1",
            "--synthetic-data",
            "host",
            "--validation",
            "False",
            "--num-replicas",
            "1",
        )

        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)

    def test_wrong_multiple_values_pipeline(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        with self.assertRaises(AssertionError):
            output = run_train(
                self,
                "--weight-updates-per-epoch",
                "1",
                "--validation",
                "False",
                "--dataset-path",
                "/localdata/datasets/",
                "--available-memory-proportion",
                "50",
                "50",
                "60",
                "--pipeline-splits",
                "conv2d_1",
                "--gradient-accumulation-count",
                "4",
            )


@pytest.mark.skip(reason="use of external data (T68092)")
class StableNormStochasticRoundingFloatingPointExceptions(SubProcessChecker):
    def test_stable_norm_stochastic_rounding_fp_exceptions_enabled(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        output = run_train(
            self,
            "--weight-updates-per-epoch",
            "1",
            "--dataset-path",
            "/localdata/datasets/",
            "--stable-norm",
            "True",
            "--stochastic-rounding",
            "ON",
            "--fp-exceptions",
            "True",
            "--norm-layer",
            '{"name": "batch_norm", "momentum": 0.0}',
            "--model",
            "cifar_resnet8",
        )
        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)

    def test_stable_norm_tochastic_rounding_fp_exceptions_disabled(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        output = run_train(
            self,
            "--weight-updates-per-epoch",
            "1",
            "--dataset-path",
            "/localdata/datasets/",
            "--stable-norm",
            "False",
            "--stochastic-rounding",
            "OFF",
            "--fp-exceptions",
            "False",
            "--norm-layer",
            '{"name": "batch_norm", "momentum": 0.0}',
            "--model",
            "cifar_resnet8",
        )
        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)


@pytest.mark.skip(reason="use of external data (T68092)")
class LRSchedules(SubProcessChecker):
    def test_cosine_lr_schedule(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        output = run_train(
            self,
            "--weight-updates-per-epoch",
            "1",
            "--dataset-path",
            "/localdata/datasets/",
            "--lr-schedule",
            "cosine",
            "--lr-schedule-params",
            '{"initial_learning_rate": 0.0001, "epochs_to_total_decay": 1.1}',
        )
        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)

    def test_stepped_lr_schedule(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        output = run_train(
            self,
            "--weight-updates-per-epoch",
            "1",
            "--dataset-path",
            "/localdata/datasets/",
            "--lr-schedule",
            "stepped",
            "--lr-schedule-params",
            '{"boundaries": [0.1, 0.5, 0.8], "values": [0.00001, 0.0001, 0.0005, 0.00001]}',
        )
        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)

    def test_const_schedule_with_shift_warmup(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        output = run_train(
            self,
            "--weight-updates-per-epoch",
            "1",
            "--dataset-path",
            "/localdata/datasets/",
            "--lr-warmup-params",
            '{"warmup_mode": "shift", "warmup_epochs": 1.1}',
        )
        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)

    def test_const_schedule_with_mask_warmup(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        output = run_train(
            self,
            "--weight-updates-per-epoch",
            "1",
            "--dataset-path",
            "/localdata/datasets/",
            "--lr-warmup-params",
            '{"warmup_mode": "mask", "warmup_epochs": 1.1}',
        )
        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)

    def test_const_schedule_with_staircase(self):
        path_to_cifar10 = "/localdata/datasets/cifar10"
        if not os.path.exists(path_to_cifar10):
            raise NameError(f"Directory {path_to_cifar10} from TFDS should have been copied to CI for this test")
        output = run_train(
            self, "--weight-updates-per-epoch", "1", "--dataset-path", "/localdata/datasets/", "--lr-staircase", "True"
        )
        self.assertIn("loss:", output)
        self.assertIn("training_accuracy:", output)
