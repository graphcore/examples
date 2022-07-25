# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import unittest
from pathlib import Path
import sys
import tensorflow as tf
import os
import shutil
import cv2
import numpy as np
import tensorflow_datasets as tfds

sys.path.append(str(Path(__file__).absolute().parent.parent))

from datasets.data_generator import DataGenerator


class UnsupportedDsTest(unittest.TestCase):

    def test_unsupported_dataset(self):
        with self.assertRaises(NameError):
            _ = DataGenerator.get_dataset_from_name(ds_name='foo')

    def test_mnist_wrong_dir(self):
        with self.assertRaises(NameError):
            _ = DataGenerator.get_dataset_from_name(
                ds_name='mnist', ds_path='foo')


class CheckSize(unittest.TestCase):
    def test_check_size(self):
        ds = tf.data.Dataset.from_tensor_slices([1, 2, 3])
        assert(DataGenerator.evaluate_size_dataset(ds) == 3)


class CreateDatasetFromSources(unittest.TestCase):

    def test_cifar10(self):
        dirpath = '/tmp/cifar10_test_sample_directory'
        if not (os.path.exists(dirpath)):
            os.makedirs(dirpath)

        ds, img_shape, num_examples, num_classes = DataGenerator.get_dataset_from_name(
            ds_name='cifar10', ds_path=dirpath, split='train[0:3]')

        assert (num_classes == 10)
        assert (num_examples == 3)
        assert (tuple(img_shape) == (32, 32, 3))
        assert (isinstance(ds, tf.data.Dataset))

        ds, img_shape, num_examples, num_classes = DataGenerator.get_dataset_from_name(
            ds_name='cifar10', ds_path=dirpath, split='test[0:3]')

        assert (num_classes == 10)
        assert (num_examples == 3)
        assert (tuple(img_shape) == (32, 32, 3))
        assert (isinstance(ds, tf.data.Dataset))

    def test_mnist(self):
        dirpath = '/tmp/mnist_test_sample_directory'
        if not (os.path.exists(dirpath)):
            os.makedirs(dirpath)

        ds, img_shape, num_examples, num_classes = DataGenerator.get_dataset_from_name(
            ds_name='mnist', ds_path=dirpath, split='train[0:3]')

        assert (num_classes == 10)
        assert (num_examples == 3)
        assert (tuple(img_shape) == (28, 28, 1))
        assert (isinstance(ds, tf.data.Dataset))

        ds, img_shape, num_examples, num_classes = DataGenerator.get_dataset_from_name(
            ds_name='mnist', ds_path=dirpath, split='test[0:3]')

        assert (num_classes == 10)
        assert (num_examples == 3)
        assert (tuple(img_shape) == (28, 28, 1))
        assert (isinstance(ds, tf.data.Dataset))


class CreateSyntheticDataset(unittest.TestCase):
    def test_synthetic_random_data(self):
        ds = DataGenerator.get_random_dataset(
            height=32, width=32, num_classes=5, data_type=tf.float32)
        assert (isinstance(ds, tf.data.Dataset))

    def test_fetch_from_directory(self):

        dirpath = '/tmp/temporary_dataset_directory'
        split_name = 'tmp_train'
        label_name1 = 'tmp_label1'
        label_name2 = 'tmp_label2'
        full_dir_to_create_1 = os.path.join(dirpath, split_name, label_name1)
        full_dir_to_create_2 = os.path.join(dirpath, split_name, label_name2)
        image_path_1 = os.path.join(
            full_dir_to_create_1, 'temporary_image_1.jpg')
        image_path_2 = os.path.join(
            full_dir_to_create_2, 'temporary_image_2.jpg')
        if not (os.path.exists(dirpath)):
            os.makedirs(full_dir_to_create_1)
            os.makedirs(full_dir_to_create_2)
            image_1 = np.ones((1, 1, 3), np.uint8)
            cv2.imwrite(image_path_1, image_1)
            image_2 = np.ones((1, 1, 3), np.uint8)
            cv2.imwrite(image_path_2, image_2)

        ds, img_shape, num_examples, num_classes = DataGenerator.get_dataset_from_directory(
            ds_path=dirpath, split=split_name)

        assert (num_classes == 2)
        assert (num_examples == 2)
        assert (tuple(img_shape) == (1, 1, 3))
        assert (isinstance(ds, tf.data.Dataset))

        for image, label in tfds.as_numpy(ds):
            assert(np.all(image == 1))
            assert (isinstance(label, np.int64))

        shutil.rmtree(dirpath)


class LoadGCImagenetTest(unittest.TestCase):

    def test_load_imagenet_train(self):

        imagenet_tf_record_data_path = '/localdata/datasets/imagenet-data'
        # if the path doesn't exist the function will throw an error
        ds, img_shape, num_examples, num_classes = DataGenerator.get_imagenet(
            imagenet_tf_record_data_path, 'train', seed=None)

        computed_size = DataGenerator.evaluate_size_dataset(ds)

        self.assertTrue(isinstance(ds, tf.data.Dataset))
        self.assertEqual(img_shape, (224, 224, 3))
        self.assertEqual(num_examples, computed_size)
        self.assertEqual(num_examples, 1281167)
        self.assertEqual(num_classes, 1000)

    def test_load_imagenet_validation(self):

        imagenet_tf_record_data_path = '/localdata/datasets/imagenet-data'

        # if the path doesn't exist the function will throw an error
        ds, img_shape, num_examples, num_classes = DataGenerator.get_imagenet(
            imagenet_tf_record_data_path, 'validation', seed=None)

        computed_size = DataGenerator.evaluate_size_dataset(ds)

        self.assertTrue(isinstance(ds, tf.data.Dataset))
        self.assertEqual(img_shape, (224, 224, 3))
        self.assertEqual(num_examples, 50000)
        self.assertEqual(num_examples, computed_size)
        self.assertEqual(num_classes, 1000)


class BuildImagenetTest(unittest.TestCase):

    def test_malformed_directory(self):
        imagenet_original_data_path = '/localdata/datasets/imagenet-raw-data'
        output_directory = '/tmp/temporary_imagenet_dataset_directory'
        split = 'val'

        with self.assertRaises(ValueError):
            # if the path imagenet_original_data_path does not contain 1000 directories the function will throw an error
            DataGenerator.build_imagenet_tf_record(imagenet_original_data_path,
                                                   split, output_directory=output_directory)

    def test_build_imagenet_validation(self):

        imagenet_original_data_path = '/localdata/datasets/imagenet-raw-data/validation'
        output_directory = '/tmp/temporary_imagenet_dataset_directory'
        split = 'validation'

        DataGenerator.build_imagenet_tf_record(imagenet_original_data_path,
                                               split, output_directory=output_directory)

        ds, img_shape, num_examples, num_classes = DataGenerator.get_imagenet(
            output_directory, split, seed=None)

        computed_size = DataGenerator.evaluate_size_dataset(ds)

        self.assertTrue(isinstance(ds, tf.data.Dataset))
        self.assertEqual(img_shape, (224, 224, 3))
        self.assertEqual(num_examples, 50000)
        self.assertEqual(num_examples, computed_size)
        self.assertEqual(num_classes, 1000)

        shutil.rmtree(output_directory)
