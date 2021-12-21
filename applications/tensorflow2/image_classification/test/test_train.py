# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import unittest
import subprocess
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))
from test_common import run_train


class WrongArguments(unittest.TestCase):
    def test_unsupported_argument(self):
        with self.assertRaises(subprocess.CalledProcessError) as e:
            run_train('--wrong-arg', '0')
        self.assertEqual(e.exception.returncode, 2)

    def test_wrong_device_assignment(self):
        with self.assertRaises(subprocess.CalledProcessError) as e:
            run_train('--model-name', 'cifar_resnet8',
                      '--pipeline-splits', 'conv2_block1_post_addition_relu', 'conv3_block1_post_addition_relu', 'conv4_block1_post_addition_relu',
                      '--device-mapping', '0', '1', '2', '4',
                      '--gradient-accumulation-count', '8',
                      '--recomputation', 'True')
        self.assertEqual(e.exception.returncode, 1)

    def test_recomputation_no_pipeline(self):
        with self.assertRaises(subprocess.CalledProcessError) as e:
            run_train('--weight-updates-per-epoch', '1',
                      '--dataset', 'imagenet',
                      '--dataset-path', '/localdata/datasets/imagenet-data',
                      '--accelerator-side-preprocess', 'True',
                      '--gradient-accumulation-count', '8',
                      '--recomputation', 'True')

    def test_recompute_with_no_split(self):
        with self.assertRaises(subprocess.CalledProcessError) as e:
            run_train('--recomputation', 'True')
        self.assertEqual(e.exception.returncode, 1)

    def test_logs_per_epoch_neg(self):
        with self.assertRaises(subprocess.CalledProcessError) as e:
            run_train('--logs-per-epoch', '-1')
        self.assertEqual(e.exception.returncode, 1)

    def test_logs_per_epoch_not_multiple_of_epoch(self):
        with self.assertRaises(subprocess.CalledProcessError) as e:
            run_train('--logs-per-epoch', '0.25', '--num-epochs', '6')
        self.assertEqual(e.exception.returncode, 1)

    # Must be updated when supported
    def test_cifar_resnet_device_not_supported(self):
        with self.assertRaises(subprocess.CalledProcessError) as e:
            run_train('--model', 'cifar_resnet8',
                      '--weight-updates-per-epoch', '1',
                      '--accelerator-side-preprocess', 'True',
                      '--eight-bit-transfer', 'True')
        self.assertEqual(e.exception.returncode, 1)

    def test_non_accelerated_8_bit_io(self):
        with self.assertRaises(subprocess.CalledProcessError) as e:
            run_train('--accelerator-side-preprocess', 'False',
                      '--micro-batch-size', '8', '--validation',
                      '--eight-bit-transfer', 'True')
        self.assertEqual(e.exception.returncode, 1)

# These tests are likely


class SimplePass(unittest.TestCase):
    def test_help(self):
        output = run_train('--help')
        self.assertIn('usage', output)

    def test_one_update_mnist(self):
        path_to_mnist = '/localdata/datasets/mnist'
        if not os.path.exists(path_to_mnist):
            raise NameError(f'Directory {path_to_mnist} from TFDS should have been copied to CI for this test')
        output = run_train('--weight-updates-per-epoch', '1',
                           '--dataset', 'mnist',
                           '--label-smoothing', '0.1',
                           '--dataset-path', '/localdata/datasets/')
        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)

    def test_one_update_accelerator(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        output = run_train('--weight-updates-per-epoch', '1',
                           '--accelerator-side-preprocess', 'True',
                           '--eight-bit-transfer', 'True',
                           '--label-smoothing', '0.1',
                           '--dataset-path', '/localdata/datasets/')
        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)


class Resnet8(unittest.TestCase):
    def test_config(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        output = run_train('--weight-updates-per-epoch', '1', '--config', 'resnet8_test')
        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)

    def test_logs_per_epoch_lt_one(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        output = run_train('--model', 'cifar_resnet8',
                           '--dataset-path', '/localdata/datasets/',
                           '--num-epochs', '2',
                           '--precision', '16.16',
                           '--micro-batch-size', '8',
                           '--validation', 'False',
                           '--half-partials', 'True',
                           '--gradient-accumulation', '8',
                           '--logs-per-epoch', '0.5')
        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)

    def test_pipeline(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        output = run_train('--model-name', 'cifar_resnet8',
                           '--dataset-path', '/localdata/datasets/',
                           '--pipeline-splits', 'conv2_block1_post_addition_relu', 'conv3_block1_post_addition_relu', 'conv4_block1_post_addition_relu',
                           '--device-mapping', '0', '1', '2', '3',
                           '--gradient-accumulation-count', '8',
                           '--recomputation', 'True')
        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)


class ImageNet(unittest.TestCase):
    def test_accelerated(self):
        path_to_imagenet = '/localdata/datasets/imagenet-data'
        if not os.path.exists(path_to_imagenet):
            raise NameError(f'Directory {path_to_imagenet} should have been copied to CI for this test')
        output = run_train('--weight-updates-per-epoch', '1',
                           '--dataset', 'imagenet',
                           '--dataset-path', '/localdata/datasets/imagenet-data',
                           '--accelerator-side-preprocess', 'True',
                           '--micro-batch-size', '8',
                           '--validation', 'False',
                           '--half-partials', 'True',
                           '--gradient-accumulation', '8',
                           '--eight-bit-transfer', 'True')
        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)

    def test_non_accelerated(self):
        path_to_imagenet = '/localdata/datasets/imagenet-data'
        if not os.path.exists(path_to_imagenet):
            raise NameError(f'Directory {path_to_imagenet} should have been copied to CI for this test')
        output = run_train('--weight-updates-per-epoch', '1',
                           '--dataset', 'imagenet',
                           '--dataset-path', '/localdata/datasets/imagenet-data',
                           '--accelerator-side-preprocess', 'False',
                           '--micro-batch-size', '8',
                           '--validation', 'False',
                           '--half-partials', 'True',
                           '--precision', '16.16',
                           '--gradient-accumulation', '8')
        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)


class DisableVariableOffloading(unittest.TestCase):

    def test_disable_variable_offloading_global_batch_size(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        output = run_train('--weight-updates-per-epoch', '1',
                           '--dataset-path', '/localdata/datasets/',
                           '--global-batch-size', '10')
        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)

    def test_both_args(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        with self.assertRaises(subprocess.CalledProcessError) as e:
            output = run_train('--weight-updates-per-epoch', '1',
                               '--dataset-path', '/localdata/datasets/',
                               '--gradient-accumulation-count', '10',
                               '--global-batch-size', '10')
        self.assertEqual(e.exception.returncode, 1)


class AvailableMemoryProportion(unittest.TestCase):

    def test_single_value_pipeline(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        output = run_train('--weight-updates-per-epoch', '1',
                           '--validation', 'False',
                           '--dataset-path', '/localdata/datasets/',
                           '--available-memory-proportion', '50',
                           '--pipeline-splits', 'conv2d_1',
                           '--gradient-accumulation-count', '4')

        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)

    def test_multiple_values_no_pipelining(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        with self.assertRaises(subprocess.CalledProcessError) as e:
            output = run_train('--weight-updates-per-epoch', '1',
                               '--validation', 'False',
                               '--dataset-path', '/localdata/datasets/',
                               '--available-memory-proportion', '50', '50')

        self.assertEqual(e.exception.returncode, 1)

    def test_multiple_values_pipeline(self):

        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        output = run_train('--weight-updates-per-epoch', '1',
                           '--validation', 'False',
                           '--dataset-path', '/localdata/datasets/',
                           '--available-memory-proportion', '50', '50', '60', '60',
                           '--pipeline-splits', 'conv2d_1',
                           '--gradient-accumulation-count', '4')

        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)

    def test_pipeline_mixed_precision_resnet50(self):

        path_to_imagenet = '/localdata/datasets/imagenet-data'
        if not os.path.exists(path_to_imagenet):
            raise NameError(f'Directory {path_to_imagenet} should have been copied to CI for this test')
        output = run_train('--config', 'mk2_resnet50_16k_bn_pipeline',
                           '--weight-updates-per-epoch', '1',
                           '--num-epochs', '1',
                           '--micro-batch-size', '1',
                           '--label-smoothing', '0.1',
                           '--validation', 'False', '--num-replicas', '1')

        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)

    def test_wrong_multiple_values_pipeline(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        with self.assertRaises(subprocess.CalledProcessError) as e:
            output = run_train('--weight-updates-per-epoch', '1',
                               '--validation', 'False',
                               '--dataset-path', '/localdata/datasets/',
                               '--available-memory-proportion', '50', '50', '60',
                               '--pipeline-splits', 'conv2d_1',
                               '--gradient-accumulation-count', '4')

        self.assertEqual(e.exception.returncode, 1)


class StableNormStochasticRoundingFloatingPointExceptions(unittest.TestCase):

    def test_stable_norm_stochastic_rounding_fp_exceptions_enabled(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        output = run_train('--weight-updates-per-epoch', '1',
                           '--dataset-path', '/localdata/datasets/',
                           '--stable-norm', 'True',
                           '--stochastic-rounding', 'True',
                           '--fp-exceptions', 'True',
                           '--bn-momentum', '0.0',
                           '--model', 'cifar_resnet8')
        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)

    def test_stable_norm_tochastic_rounding_fp_exceptions_disabled(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        output = run_train('--weight-updates-per-epoch', '1',
                           '--dataset-path', '/localdata/datasets/',
                           '--stable-norm', 'False',
                           '--stochastic-rounding', 'False',
                           '--fp-exceptions', 'False',
                           '--bn-momentum', '0.0',
                           '--model', 'cifar_resnet8')
        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)


class LRSchedules(unittest.TestCase):

    def test_cosine_lr_schedule(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        output = run_train('--weight-updates-per-epoch', '1',
                           '--dataset-path', '/localdata/datasets/',
                           '--lr-schedule', 'cosine',
                           '--lr-schedule-params', '{"initial_learning_rate": 0.0001, "epochs_to_total_decay": 1.1}')
        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)

    def test_stepped_lr_schedule(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        output = run_train('--weight-updates-per-epoch', '1',
                           '--dataset-path', '/localdata/datasets/',
                           '--lr-schedule', 'stepped',
                           '--lr-schedule-params', '{"boundaries": [0.1, 0.5, 0.8], "values": [0.00001, 0.0001, 0.0005, 0.00001]}')
        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)

    def test_const_schedule_with_shift_warmup(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        output = run_train('--weight-updates-per-epoch', '1',
                           '--dataset-path', '/localdata/datasets/',
                           '--lr-warmup-params', '{"warmup_mode": "shift", "warmup_epochs": 1.1}')
        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)

    def test_const_schedule_with_mask_warmup(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        output = run_train('--weight-updates-per-epoch', '1',
                           '--dataset-path', '/localdata/datasets/',
                           '--lr-warmup-params', '{"warmup_mode": "mask", "warmup_epochs": 1.1}')
        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)

    def test_const_schedule_with_staircase(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        output = run_train('--weight-updates-per-epoch', '1',
                           '--dataset-path', '/localdata/datasets/',
                           '--lr-staircase', 'True')
        self.assertIn('loss:', output)
        self.assertIn('training_accuracy:', output)
