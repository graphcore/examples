# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import unittest
import tensorflow as tf

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))

from datasets.cifar10_dataset import CIFAR10Dataset
from datasets.application_dataset import ApplicationDataset

class LoadCIFAR10(unittest.TestCase):

    def test_cifar10_train_split(self):

        cifar10_dataset = CIFAR10Dataset(
            dataset_name='cifar10',
            dataset_path=str(Path(__file__).absolute().parent.parent),
            split='train'
        )

        pipeline = cifar10_dataset.read_single_image()

        assert (cifar10_dataset.num_classes() == 10)
        assert (cifar10_dataset.size() == 50000)
        assert (cifar10_dataset.image_shape() == (32, 32, 3))
        assert (isinstance(pipeline, tf.data.Dataset))

        app_dataset = ApplicationDataset(
            pipeline,
            cifar10_dataset.size(),
            cifar10_dataset.image_shape(),
            cifar10_dataset.num_classes
        )

        assert(cifar10_dataset.size() == app_dataset.evaluate_size(micro_batch_size=1))

    def test_cifar10_validation_split(self):

        cifar10_dataset = CIFAR10Dataset(
            dataset_name='cifar10',
            dataset_path=str(Path(__file__).absolute().parent.parent),
            split='test'
        )

        pipeline = cifar10_dataset.read_single_image()

        assert (cifar10_dataset.num_classes() == 10)
        assert (cifar10_dataset.size() == 10000)
        assert (cifar10_dataset.image_shape() == (32, 32, 3))
        assert (isinstance(pipeline, tf.data.Dataset))

        app_dataset = ApplicationDataset(
            pipeline,
            cifar10_dataset.size(),
            cifar10_dataset.image_shape(),
            cifar10_dataset.num_classes
        )

        assert(cifar10_dataset.size() == app_dataset.evaluate_size(micro_batch_size=1))
