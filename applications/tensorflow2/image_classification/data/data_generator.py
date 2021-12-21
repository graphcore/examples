# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import tensorflow_datasets as tfds
import os
import logging
import popdist

from custom_exceptions import UnsupportedFormat, DimensionError
from . import imagenet_processing
from . import build_imagenet_data
import glob


AVAILABLE_DATASET = {'mnist', 'cifar10', 'cifar100', 'imagenet'}
IMAGENET_DS_SIZE = {'train': 1281167, 'test': 50000, 'validation': 50000}


class DataGenerator:

    logger = logging.getLogger('data_generator')

    @staticmethod
    def get_dataset_from_name(
            ds_name: str,
            ds_path: str = None,
            split: str = 'train', seed=42):

        if ds_path is not None:
            if not os.path.exists(ds_path):
                DataGenerator.logger.error(f'{ds_path} does not exist')
                raise NameError(f'Directory {ds_path} does not exist')

        if ds_name not in AVAILABLE_DATASET:
            DataGenerator.logger.info(
                f'Dataset name {ds_name} is not supported. Supported dataset: {list(AVAILABLE_DATASET)}')
            raise NameError(
                f'Dataset name {ds_name} is not supported. Supported dataset: {list(AVAILABLE_DATASET)}'
            )

        if ds_name == 'imagenet':
            return DataGenerator.get_imagenet(ds_path, split)

        else:
            ds, info_ds = tfds.load(
                ds_name,
                data_dir=ds_path,
                split=split,
                # If true, returns `(img, label)` instead of dict(image=, ...)
                as_supervised=True,
                with_info=True
            )

            if not isinstance(ds, tf.data.Dataset):
                raise UnsupportedFormat(
                    f'Type of ds is not the one expected (tf.data.Dataset) {type(ds)}')

            num_examples = info_ds.splits[split].num_examples

            iterator = iter(ds)
            first_elem = iterator.get_next()

            if len(first_elem[0].shape) != 3:
                raise DimensionError(
                    f'Dataset input feature should have at least 3 dimensions (h,w,c) but it has {len(first_elem[0].shape)}')

            img_shape = first_elem[0].shape

            num_classes = -1

            if len(info_ds.supervised_keys) == 2:
                label = info_ds.supervised_keys[1]
                num_classes = info_ds.features[label].num_classes
            else:
                raise UnsupportedFormat(
                    f'This function only handle datasets like (features, labels) not {info_ds.supervised_keys}')

            return ds, img_shape, num_examples, num_classes

    @staticmethod
    def get_dataset_from_directory(
            ds_path: str,
            split: str, seed=42):

        if not os.path.exists(ds_path):
            DataGenerator.logger.error(f'{ds_path} does not exist')
            raise NameError(f'Directory {ds_path} does not exist')

        builder = tfds.folder_dataset.ImageFolder(ds_path)

        info_ds = builder.info

        ds = builder.as_dataset(as_supervised=True, split=split)

        if not isinstance(ds, tf.data.Dataset):
            raise UnsupportedFormat(
                f'Type of ds is not the one expected (tf.data.Dataset) {type(ds)}')

        num_examples = DataGenerator.evaluate_size_dataset(ds)

        iterator = iter(ds)
        first_elem = iterator.get_next()

        if len(first_elem[0].shape) != 3:
            raise DimensionError(
                f'Dataset input feature should have at least 3 dimensions (h,w,c) but it has {len(first_elem[0].shape)}')

        img_shape = first_elem[0].shape

        num_classes = -1

        if len(info_ds.supervised_keys) == 2:
            label = info_ds.supervised_keys[1]
            num_classes = info_ds.features[label].num_classes
        else:
            raise UnsupportedFormat(
                f'This function only handle datasets like (features, labels) not {info_ds.supervised_keys}')

        print(
            f'img shape {img_shape} number of examples {num_examples} number of classes {num_classes}')

        if popdist.getNumInstances() > 1:
            ds = ds.shard(num_shards=popdist.getNumInstances(), index=popdist.getInstanceIndex())

        return ds, img_shape, num_examples, num_classes

    @staticmethod
    def evaluate_size_dataset(ds):
        cnt = 0
        for _ in ds:
            cnt += 1
        return cnt

    @staticmethod
    def get_random_dataset(
            height: int,
            width: int,
            num_classes: int,
            data_type: tf.dtypes.DType = tf.float32):
        images: tf.Tensor = tf.random.truncated_normal([height, width, 3],
                                                       dtype=data_type,
                                                       mean=127,
                                                       stddev=60,
                                                       name='generated_inputs')
        labels = tf.random.uniform([],
                                   minval=0,
                                   maxval=num_classes - 1,
                                   dtype=data_type,
                                   name='generated_labels')
        ds = tf.data.Dataset.from_tensors((images, labels))

        return ds

    @staticmethod
    def get_imagenet(path: str, split: str, cycle_length: int = 4, block_length: int = 4):

        # The path is the one of dataset under TFRecord format
        if not os.path.exists(path):
            DataGenerator.logger.error(f'{path} does not exist')
            raise NameError(f'Directory {path} does not exist')

        if split == 'train':
            filenames = glob.glob1(path, 'train*')
            if len(filenames) != 1024:
                DataGenerator.logger.error(
                    f'train directory should contain 1024 tf-record files but it contains {len(filenames)} instead')
                raise ValueError(f'train directory should contain 1024 files but it contains {len(filenames)} instead')

        else:
            filenames = glob.glob1(path, 'validation*')
            if len(filenames) != 128:
                DataGenerator.logger.error(
                    f'validation directory should contain 128 tf-record files but it contains {len(filenames)} instead')
                raise ValueError(
                    f'validation directory should contain 128 tf-record files but it contains {len(filenames)} instead')

        num_files = len(filenames)

        filenames = list(
            map(lambda filename: os.path.join(path, filename), filenames))
        DataGenerator.logger.debug(f'filenames = {filenames}')
        ds = tf.data.Dataset.from_tensor_slices(filenames)

        if split == 'train':
            # Shuffle the input files
            ds = ds.shuffle(buffer_size=num_files)

        if popdist.getNumInstances() > 1:
            ds = ds.shard(num_shards=popdist.getNumInstances(), index=popdist.getInstanceIndex())

        ds = ds.interleave(tf.data.TFRecordDataset,
                           cycle_length=cycle_length,
                           block_length=block_length,
                           num_parallel_calls=cycle_length)

        DataGenerator.logger.info(f'dataset = {ds}')

        num_examples = IMAGENET_DS_SIZE[split]

        DataGenerator.logger.info(f'number of examples {num_examples}')

        iterator = iter(ds)
        first_elem = iterator.get_next()

        feature, _ = imagenet_processing.parse_record(
            first_elem, True, tf.float32)

        if len(feature.shape) != 3:
            raise DimensionError(
                f'Dataset input feature should have at least 3 dimensions (h,w,c) but it has {len(first_elem[0].shape)}')

        num_classes = 1000
        ds = ds.cache()

        return ds, feature.shape, num_examples, num_classes

    @staticmethod
    def build_imagenet_tf_record(dataset_path: str, split: str, num_threads: int = 32, output_directory: str = '.', imagenet_metadata_file='imagenet_metadata.txt', bounding_box_file='imagenet_2012_bounding_boxes.csv'):

        # The path to the class directories should be constructed from dataset_path+split
        # Each class directory has the name of a class
        # For the imagenet train and validation the num_shards respectively equals to on 1024 and 128.
        if split == 'train':
            num_shards = 1024
        else:
            num_shards = 128

        full_path_to_original = os.path.join(dataset_path, split)
        full_path_to_metadata = os.path.join(dataset_path, imagenet_metadata_file)
        full_path_to_bounding_boxes = os.path.join(dataset_path, bounding_box_file)

        if not os.path.exists(full_path_to_original):
            DataGenerator.logger.error(f'Directory {dataset_path}/{split}/ does not exist')
            raise NameError(f'Directory {dataset_path}/{split}/ does not exist')

        if not os.path.exists(full_path_to_metadata):
            DataGenerator.logger.error(
                f'File {imagenet_metadata_file} does not exist and should be in the root directory {dataset_path}')
            raise NameError(
                f'File {imagenet_metadata_file} does not exist and should be in the root directory {dataset_path}')

        if not os.path.exists(full_path_to_bounding_boxes):
            DataGenerator.logger.error(
                f'File {bounding_box_file} file does not exist and should be in the root directory {dataset_path}')
            raise NameError(
                f'File {bounding_box_file} file does not exist and should be in the root directory {dataset_path}')

        # The output directory should have been created before
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        synset_to_human = build_imagenet_data._build_synset_lookup(full_path_to_metadata)
        image_to_bboxes = build_imagenet_data._build_bounding_box_lookup(full_path_to_bounding_boxes)

        build_imagenet_data._process_dataset(split, full_path_to_original, num_shards,
                                             num_threads, synset_to_human, image_to_bboxes, output_directory)

        filenames = glob.glob1(output_directory, split + '*')

        if len(filenames) != num_shards:
            DataGenerator.logger.error(
                f'File {output_directory} should contain  {num_shards} records but it contains {len(filenames)}')
            raise ValueError(
                f'File {output_directory} should contain  {num_shards} records but it contains {len(filenames)}')

        return filenames
