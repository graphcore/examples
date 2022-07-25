# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from test_common import run_train
import os
import glob
import subprocess
from examples_tests.test_util import SubProcessChecker


class Simple(SubProcessChecker):
    def test_simple_validation(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        output = run_train(self, '--weight-updates-per-epoch', '1',
                           '--dataset-path', '/localdata/datasets/',
                           '--training', 'False')
        self.assertIn('loss:', output)
        self.assertIn('validation_accuracy:', output)

    def test_empty_checkpoint_directory(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')

        checkpoint_dir = '/tmp/checkpoint_test_empty_dir'

        output = run_train(self, '--weight-updates-per-epoch', '1',
                           '--dataset-path', '/localdata/datasets/',
                           '--checkpoints', 'True',
                           '--checkpoint-dir', checkpoint_dir,
                           '--training', 'False')

        self.assertIn('loss:', output)
        self.assertIn('validation_accuracy:', output)

    def test_no_checkpoint(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')

        output = run_train(self, '--weight-updates-per-epoch', '1',
                           '--dataset-path', '/localdata/datasets/',
                           '--checkpoints', 'False')

        self.assertIn('loss:', output)
        self.assertIn('validation_accuracy:', output)


class Checkpoint(SubProcessChecker):
    def test_validation_on_ckpt(self):
        path_to_cifar10 = '/localdata/datasets/cifar10'
        checkpoint_dir = '/tmp/checkpoint_test_validation'
        if not os.path.exists(path_to_cifar10):
            raise NameError(f'Directory {path_to_cifar10} from TFDS should have been copied to CI for this test')
        output = run_train(self, '--weight-updates-per-epoch', '1',
                           '--dataset-path', '/localdata/datasets/',
                           '--checkpoints', 'True',
                           '--checkpoint-dir', checkpoint_dir,
                           '--clean-dir', 'False')
        self.assertIn('loss:', output)
        self.assertIn('validation_accuracy:', output)
        list_ckpt = glob.glob(os.path.join(checkpoint_dir, '*.h5'))
        assert(len(list_ckpt) == 1)
        output = run_train(self, '--weight-updates-per-epoch', '1',
                           '--dataset-path', '/localdata/datasets/',
                           '--checkpoints', 'True',
                           '--checkpoint-dir', checkpoint_dir,
                           '--training', 'False')
        self.assertIn('loss:', output)
        self.assertIn('validation_accuracy:', output)


class Resnet50(SubProcessChecker):
    def test_mixed_precision_resnet50(self):
        path_to_imagenet = '/localdata/datasets/imagenet-data'
        if not os.path.exists(path_to_imagenet):
            raise NameError(f'Directory {path_to_imagenet} should have been copied to CI for this test')

        output = run_train(self, '--config', 'resnet50_16ipus_16k_bn_pipeline',
                           '--training', 'False', '--num-replicas', '1')

        self.assertIn('loss:', output)
        self.assertIn('validation_accuracy:', output)
