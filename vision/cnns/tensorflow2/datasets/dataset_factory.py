# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import logging
import os
from typing import Callable, Optional, Tuple

import popdist
import psutil
import tensorflow as tf
from eight_bit_transfer import EightBitTransfer

from . import (application_dataset, cifar10_dataset, imagenet_dataset,
               mnist_dataset)
from batch_config import BatchConfig

PREFETCH_BUFFER_SIZE = 16
AVAILABLE_DATASETS = ['imagenet', 'cifar10', 'mnist']


class DatasetFactory:

    @staticmethod
    def get_dataset(dataset_name: str,
                    dataset_path: str,
                    split: str,
                    img_datatype: tf.dtypes.DType,
                    batch_config: BatchConfig,
                    seed: Optional[int] = None,
                    shuffle: bool = False,
                    deterministic: bool = False,
                    accelerator_side_preprocess: bool = True,
                    apply_preprocessing: bool = True,
                    pipeline_num_parallel: int = 48,
                    num_local_instances: int = 1,
                    fused_preprocessing: bool = False,
                    synthetic_data: Optional[str] = None,
                    eight_bit_transfer: Optional[EightBitTransfer] = None
                    ) -> Tuple[application_dataset.ApplicationDataset, Optional[Callable], int]:
        """Creates a dataset pipeline where preprocessing is divided on the host- and ipu-side.

        Args:
            dataset_name (str):
                        Dataset to use.
            dataset_path (str):
                        Path to the root folder of the dataset.
            split (str):
                        Each dataset has its own split names.
                        Usually either 'train', 'test' or 'validation'.
            img_datatype (tf.dtypes.DType):
                        Image format.
            micro_batch_size (int):
                        Number of samples that the dataset pipeline will output for each call.
            seed (str):
                        Seeding for the prng.
            shuffle (bool):
                        Enables shuffling.
                        Some datasets shuffle both files and samples e.g. ImageNet.
            deterministic (bool):
                        Sets dataset.map() and .interleave()
                        to behave deterministically.
            accelerator_side_preprocess (bool):
                        Dataset-specific. Makes parts of the preprocessing
                        to be performed on the IPUs rather than the host.
            apply_preprocessing (bool):
                        Allows to disable preprocessing completely.
            pipeline_num_parallel (int):
                        Number of parallel threads to be used for the preprocessing.
            num_local_instances (int):
                        Number of instances running on a single host machine.
            fused_preprocessing (bool):
                        An optimisation allowing better memory mapping in some cases.
            synthetic_data (str):
                        Used for benchamarking and tput evaluation. Can be either 'cpu' or
                        'ipu'. An artificial dataset is generated on either sides to measure
                        bottlenecks in the processing.
            eight_bit_transfer (EightBitTransfer):
                        If enabled, the data streamed from the host to the IPUs
                        is in uint8 rather than the original type.

        Returns:
            Tuple[
                application_dataset.ApplicationDataset: host-side dataset pipeline
                Optional[Callable]: part of the preprocessing run on the ipu rather than host
                int: number of parallel threads for the host-side dataset pipeline
            ]
        """

        logging.info(f'dataset_name = {dataset_name}')

        max_threads_per_local_instances = os.cpu_count() // num_local_instances

        if pipeline_num_parallel > max_threads_per_local_instances:
            # Limit the maximal number of threads to the total of
            # physical threads divided by the number of instances
            logging.warning(f'The number of chosen threads {pipeline_num_parallel} is bigger than '
                            'the total number of physical threads divided by the local number of '
                            'instances. Poprun will override the config. ')
            pipeline_num_parallel = max_threads_per_local_instances

        if popdist.getNumInstances() == 1 and not deterministic:
            logging.info('Since the training is run in a single process, setting dataset pipeline '
                         'threading and prefetching buffer size to tf.data.AUTOTUNE.')
            pipeline_num_parallel = prefetch_size = tf.data.AUTOTUNE
        else:
            prefetch_size = PREFETCH_BUFFER_SIZE
            logging.info(f'Setting number of threads for the dataset pipeline to {pipeline_num_parallel}, '
                         f'and the prefetching buffer size to {prefetch_size}.')

        if dataset_name == 'imagenet':
            dataset = imagenet_dataset.ImagenetDataset(
                dataset_path=dataset_path,
                split=split,
                shuffle=shuffle,
                deterministic=deterministic,
                seed=seed,
                img_datatype=img_datatype,
                accelerator_side_preprocess=accelerator_side_preprocess,
                fused_preprocessing=fused_preprocessing
            )

        elif dataset_name == 'cifar10':
            dataset = cifar10_dataset.CIFAR10Dataset(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                split=split,
                shuffle=shuffle,
                deterministic=deterministic,
                seed=seed,
                img_datatype=img_datatype,
                accelerator_side_preprocess=accelerator_side_preprocess
            )

        elif dataset_name == 'mnist':
            dataset = mnist_dataset.MNISTDataset(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                split=split,
                shuffle=shuffle,
                deterministic=deterministic,
                seed=seed,
                img_datatype=img_datatype,
                accelerator_side_preprocess=accelerator_side_preprocess
            )

        else:
            raise ValueError(f'Unknown dataset {dataset_name}')

        app_dataset = dataset.read_single_image()
        ds = app_dataset.pipeline

        ipu_preprocess_fn = None
        if apply_preprocessing:
            cpu_preprocessing_fn = dataset.cpu_preprocessing_fn()
            ipu_preprocess_fn = (
                dataset.ipu_preprocessing_fn() if accelerator_side_preprocess else None)

            if cpu_preprocessing_fn is not None:
                ds = ds.map(cpu_preprocessing_fn,
                            num_parallel_calls=pipeline_num_parallel,
                            deterministic=deterministic)

        ds = dataset.post_preprocessing_pipeline(ds)

        if split != 'train':
            num_discarded_samples = batch_config.get_num_discarded_samples_per_instance(app_dataset.size,
                                                                                        popdist.getNumInstances())
            if num_discarded_samples > 0:
                num_padding_samples = batch_config.get_num_padding_samples_per_instance(popdist.getNumInstances(),
                                                                                        num_discarded_samples)
                padding_dataset = DatasetFactory.get_validation_padding_dataset(app_dataset.image_shape,
                                                                                img_datatype,
                                                                                app_dataset.num_classes,
                                                                                num_padding_samples)
                ds = ds.concatenate(padding_dataset)

        if eight_bit_transfer is not None:
            ds = ds.map(lambda x, y: (eight_bit_transfer.compress(x), y),
                        num_parallel_calls=pipeline_num_parallel,
                        deterministic=deterministic)

        if synthetic_data is not None:
            if synthetic_data == "host":
                logging.info(f'Activating synthetic data on the host.')
                ds = DatasetFactory.get_synthetic_dataset(
                    height=app_dataset.image_shape[0],
                    width=app_dataset.image_shape[1],
                    num_classes=app_dataset.num_classes,
                    data_type=img_datatype,
                    eight_bit_transfer=(eight_bit_transfer is not None))

            elif synthetic_data == "ipu":
                logging.info(f'Activating synthetic data on the ipu.')
                tf_poplar_flags = ' --use_synthetic_data --synthetic_data_initializer=random'
                if "TF_POPLAR_FLAGS" in os.environ:
                    os.environ["TF_POPLAR_FLAGS"] += tf_poplar_flags
                else:
                    os.environ["TF_POPLAR_FLAGS"] = tf_poplar_flags

            else:
                logging.warning(
                    f'Synthetic data option \'{synthetic_data}\' not recognized, using normal dataset')

        ds = ds.batch(batch_size=batch_config.micro_batch_size, drop_remainder=True)
        ds = ds.repeat().prefetch(prefetch_size)

        app_dataset.pipeline = ds

        cpu_memory_usage = psutil.virtual_memory().percent

        if cpu_memory_usage > 100:
            logging.warning(
                f'cpu_memory_usage is {cpu_memory_usage} > 100% so your program is likely to crash')

        return app_dataset, ipu_preprocess_fn, pipeline_num_parallel

    @staticmethod
    def get_synthetic_dataset(height: int,
                              width: int,
                              num_classes: int,
                              data_type: tf.dtypes.DType = tf.float32,
                              eight_bit_transfer: bool = False):

        images = tf.random.truncated_normal([height, width, 3],
                                            dtype=data_type,
                                            mean=127,
                                            stddev=60)
        if eight_bit_transfer:
            images = tf.cast(images, tf.uint8)

        labels = tf.random.uniform([],
                                   minval=0,
                                   maxval=num_classes - 1,
                                   dtype=data_type)

        ds = tf.data.Dataset.from_tensors((images, labels))
        ds = ds.cache()
        ds = ds.repeat()

        return ds

    @staticmethod
    def get_validation_padding_dataset(img_shape: Tuple[int, int, int],
                                       data_type: tf.dtypes.DType,
                                       num_classes: int,
                                       num_padding_samples: int) -> tf.data.Dataset:

        image = tf.zeros(img_shape, data_type)
        label = tf.constant((num_classes,), dtype=tf.float32)

        return tf.data.Dataset.from_tensors((image, label)).cache().repeat(num_padding_samples).take(num_padding_samples)
