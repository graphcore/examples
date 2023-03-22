# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import logging
import os
from typing import Callable, Optional, Tuple
import numpy as np

import popdist
from tensorflow.python.ipu import distributed
import tensorflow as tf
from eight_bit_transfer import EightBitTransfer

from . import application_dataset, cifar10_dataset, imagenet_dataset, mnist_dataset
from batch_config import BatchConfig

PREFETCH_BUFFER_SIZE = 16
AVAILABLE_DATASETS = ["imagenet", "cifar10", "mnist"]


class DatasetFactory:
    @staticmethod
    def get_dataset(
        dataset_name: str,
        dataset_path: str,
        split: str,
        img_datatype: tf.dtypes.DType,
        batch_config: BatchConfig,
        seed: Optional[int] = None,
        shuffle: bool = False,
        deterministic: bool = False,
        accelerator_side_preprocess: bool = True,
        pipeline_num_parallel: int = 48,
        fused_preprocessing: bool = False,
        synthetic_data: Optional[str] = None,
        eight_bit_transfer: Optional[EightBitTransfer] = None,
        repeat: bool = True,
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
            batch_config (BatchConfig):
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
            fused_preprocessing (bool):
                        An optimisation allowing better memory mapping in some cases.
            synthetic_data (str):
                        Used for benchmarking and tput evaluation. Can be either 'cpu' or
                        'ipu'. An artificial dataset is generated on either sides to measure
                        bottlenecks in the processing.
            eight_bit_transfer (EightBitTransfer):
                        If enabled, the data streamed from the host to the IPUs
                        is in uint8 rather than the original type.
            repeat (bool):
                        when True adds a repeat operation to the dataset pipeline

        Returns:
            Tuple[
                application_dataset.ApplicationDataset: host-side dataset pipeline
                Optional[Callable]: part of the preprocessing run on the ipu rather than host
                int: number of parallel threads for the host-side dataset pipeline
            ]
        """

        logging.info(f"dataset_name = {dataset_name}")

        if popdist.getNumInstances() == 1 and not deterministic:
            logging.info(
                "Since the training is run in a single process, setting dataset pipeline "
                "threading and prefetching buffer size to tf.data.AUTOTUNE."
            )
            pipeline_num_parallel = prefetch_size = tf.data.AUTOTUNE
        else:
            prefetch_size = PREFETCH_BUFFER_SIZE
            logging.info(
                f"Setting number of threads for the dataset pipeline to {pipeline_num_parallel}, "
                f"and the prefetching buffer size to {prefetch_size}."
            )

        if dataset_name == "imagenet":
            dataset = imagenet_dataset.ImagenetDataset(
                dataset_path=dataset_path,
                split=split,
                shuffle=shuffle,
                deterministic=deterministic,
                seed=seed,
                img_datatype=tf.uint8 if eight_bit_transfer is not None else img_datatype,
                accelerator_side_preprocess=accelerator_side_preprocess,
                fused_preprocessing=fused_preprocessing,
            )

        elif dataset_name == "cifar10":
            dataset = cifar10_dataset.CIFAR10Dataset(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                split=split,
                shuffle=shuffle,
                deterministic=deterministic,
                seed=seed,
                img_datatype=tf.uint8 if eight_bit_transfer is not None else img_datatype,
                accelerator_side_preprocess=accelerator_side_preprocess,
            )

        elif dataset_name == "mnist":
            dataset = mnist_dataset.MNISTDataset(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                split=split,
                shuffle=shuffle,
                deterministic=deterministic,
                seed=seed,
                img_datatype=tf.uint8 if eight_bit_transfer is not None else img_datatype,
                accelerator_side_preprocess=accelerator_side_preprocess,
            )

        else:
            raise ValueError(f"Unknown dataset {dataset_name}")

        image_shape = dataset.image_shape()

        if synthetic_data is not None:
            logging.info(f"Activating synthetic data on the host.")
            ds = DatasetFactory.get_synthetic_dataset(
                image_shape=image_shape,
                num_classes=dataset.num_classes(),
                data_type=img_datatype,
                eight_bit_transfer=(eight_bit_transfer is not None),
            )
            padded_dataset_size = dataset.size()
        else:
            ds = dataset.read_single_image()

            cpu_preprocessing_fn = dataset.cpu_preprocessing_fn()
            if cpu_preprocessing_fn is not None:
                ds = ds.map(cpu_preprocessing_fn, num_parallel_calls=pipeline_num_parallel, deterministic=deterministic)

            if split != "train":

                # compute per instance dataset size
                unpadded_dataset_size = (
                    dataset.size() if popdist.getNumInstances() == 1 else ds.reduce(0, lambda x, _: x + 1).numpy()
                )

                # compute per instance discarded samples
                num_discarded_samples = batch_config.get_num_discarded_samples(
                    unpadded_dataset_size, popdist.getNumInstances()
                )

                num_padding_samples = batch_config.get_num_padding_samples(
                    num_discarded_samples, popdist.getNumInstances()
                )

                # get padding samples from largest padded dataset size across all instances
                if popdist.getNumInstances() > 1:
                    padded_dataset_size = unpadded_dataset_size + num_padding_samples
                    padded_dataset_sizes = distributed.allgather(
                        tf.convert_to_tensor([padded_dataset_size], dtype=tf.int32)
                    )
                    padded_dataset_size = np.max(padded_dataset_sizes.numpy())
                    num_padding_samples = padded_dataset_size - unpadded_dataset_size

                if num_padding_samples > 0:
                    padding_dataset = DatasetFactory.get_validation_padding_dataset(
                        dataset.image_shape(),
                        tf.uint8 if eight_bit_transfer is not None else img_datatype,
                        dataset.num_classes(),
                        num_padding_samples,
                    )

                    ds = ds.concatenate(padding_dataset)

                padded_dataset_size = (unpadded_dataset_size + num_padding_samples) * popdist.getNumInstances()
            else:
                padded_dataset_size = -1

        ds = ds.batch(batch_size=batch_config.micro_batch_size, drop_remainder=True)
        if split != "train":
            ds = ds.cache()
        if repeat:
            ds = ds.repeat()
        ds.prefetch(prefetch_size)

        app_dataset = application_dataset.ApplicationDataset(
            pipeline=ds,
            image_shape=dataset.image_shape(),
            size=dataset.size(),
            num_classes=dataset.num_classes(),
            padded_size=padded_dataset_size,
        )

        ipu_preprocess_fn = dataset.ipu_preprocessing_fn() if accelerator_side_preprocess else None

        return app_dataset, ipu_preprocess_fn, pipeline_num_parallel

    @staticmethod
    def get_synthetic_dataset(
        image_shape: Tuple, num_classes: int, data_type: tf.dtypes.DType = tf.float32, eight_bit_transfer: bool = False
    ):

        images = tf.random.truncated_normal(image_shape, dtype=data_type, mean=127, stddev=60)
        if eight_bit_transfer:
            images = tf.cast(images, tf.uint8)

        labels = tf.random.uniform([], minval=0, maxval=num_classes - 1, dtype=data_type)

        ds = tf.data.Dataset.from_tensors((images, labels))
        ds = ds.cache()
        ds = ds.repeat()

        return ds

    @staticmethod
    def get_validation_padding_dataset(
        img_shape: Tuple[int, int, int], data_type: tf.dtypes.DType, num_classes: int, num_padding_samples: int
    ) -> tf.data.Dataset:

        image = tf.zeros(img_shape, data_type)
        label = tf.constant((num_classes,), dtype=tf.int32)

        return (
            tf.data.Dataset.from_tensors((image, label)).cache().repeat(num_padding_samples).take(num_padding_samples)
        )
