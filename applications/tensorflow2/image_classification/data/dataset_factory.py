# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import logging
from typing import Optional
import popdist

from data.data_generator import DataGenerator
from data.data_transformer import DataTransformer
from eight_bit_transfer import EightBitTransfer
import psutil
import os

IMAGENET_SHUFFLE_BUFFER_SIZE = 10000
PREFETCH_BUFFER_SIZE = 16


class DatasetFactory:

    @staticmethod
    def get_dataset(dataset_name: str,
                    dataset_path: str,
                    split: str,
                    img_datatype: tf.dtypes.DType,
                    micro_batch_size: int,
                    shuffle: bool = False,
                    accelerator_side_preprocess: bool = True,
                    eight_bit_transfer: Optional[EightBitTransfer] = None,
                    apply_preprocessing: bool = True,
                    pipeline_num_parallel: int = 48,
                    num_local_instances: int = 1,
                    fused_preprocessing: bool = False,
                    seed: Optional[int] = None,
                    synthetic_data: Optional[str] = None):

        logging.info(f'dataset_name = {dataset_name}')

        max_threads_per_local_instances = os.cpu_count() // num_local_instances

        if pipeline_num_parallel > max_threads_per_local_instances:
            logging.warning(f'The number of chosen threads {pipeline_num_parallel} is bigger than the total number of physical threads '
                            f'divided by the local number of instances. Poprun will override the config. ')
            # Limit the maximal number of threads to the total of physical threads divided by the number of instances
            pipeline_num_parallel = max_threads_per_local_instances

        if popdist.getNumInstances() == 1:
            logging.info(f'Since the training is run in a single process, setting dataset pipeline threading '
                         f'and prefetching buffer size to tf.data.AUTOTUNE.')
            pipeline_num_parallel = prefetch_size = tf.data.AUTOTUNE
        else:
            prefetch_size = PREFETCH_BUFFER_SIZE
            logging.info(f'Setting number of threads for the dataset pipeline to {pipeline_num_parallel}, '
                         f'and the prefetching buffer size to {prefetch_size}.')

        ds, img_shape, dataset_size, num_classes = DataGenerator.get_dataset_from_name(
            ds_name=dataset_name, ds_path=dataset_path, split=split, seed=seed)

        preprocess_fn = None
        if apply_preprocessing:
            if dataset_name == 'cifar10':
                ds, preprocess_fn = DataTransformer.cifar_preprocess(
                    ds,
                    buffer_size=dataset_size,
                    img_type=img_datatype,
                    is_training=(split == 'train'),
                    accelerator_side_preprocess=accelerator_side_preprocess,
                    pipeline_num_parallel=pipeline_num_parallel,
                    seed=seed
                )
            elif dataset_name == 'imagenet':
                ds, preprocess_fn = DataTransformer.imagenet_preprocessing(
                    ds,
                    img_type=img_datatype,
                    is_training=(split == 'train'),
                    accelerator_side_preprocess=accelerator_side_preprocess,
                    pipeline_num_parallel=pipeline_num_parallel,
                    fused_preprocessing=fused_preprocessing,
                    seed=seed
                )
                if shuffle:
                    ds = ds.shuffle(buffer_size=IMAGENET_SHUFFLE_BUFFER_SIZE)
            else:
                ds = DataTransformer.cache_shuffle(ds, buffer_size=dataset_size, shuffle=(split == 'train'))
                ds = DataTransformer.normalization(ds, img_type=img_datatype)
                preprocess_fn = None

            if eight_bit_transfer is not None:
                ds = ds.map(lambda x, y: (eight_bit_transfer.compress(x), y), num_parallel_calls=pipeline_num_parallel)

            if synthetic_data is not None:
                if synthetic_data == "host":
                    logging.info(f'Activating synthetic data on the host')
                    ds = DataGenerator.get_random_dataset(
                        height=img_shape[0], width=img_shape[1], num_classes=num_classes, data_type=img_datatype, eight_bit_transfer=(eight_bit_transfer is not None))
                elif synthetic_data == "ipu":
                    logging.info(f'Activating synthetic data on the ipu')
                    if "TF_POPLAR_FLAGS" in os.environ:
                        os.environ["TF_POPLAR_FLAGS"] += " --use_synthetic_data --synthetic_data_initializer=random"
                    else:
                        os.environ["TF_POPLAR_FLAGS"] = "--use_synthetic_data --synthetic_data_initializer=random"
                else:
                    logging.warning(f'Synthetic data option \'{synthetic_data}\' not recognized, using normal dataset')

            ds = ds.batch(batch_size=micro_batch_size, drop_remainder=True)
            ds = ds.repeat().prefetch(prefetch_size)

            cpu_memory_usage = psutil.virtual_memory().percent

            if cpu_memory_usage > 100:
                logging.warning(f'cpu_memory_usage is {cpu_memory_usage} > 100% so your program is likely to crash')

        return ds, img_shape, dataset_size, num_classes, preprocess_fn, pipeline_num_parallel
