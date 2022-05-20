# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import argparse
import logging

from data.dataset_factory import DatasetFactory
from data.data_generator import DataGenerator


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    # add arg parser
    def add_arguments(parser):
        parser.add_argument('--dataset', type=str, default='cifar10', help='Name of dataset to use')
        parser.add_argument('--dataset-path', type=str, default='.', help='Path to dataset')
        return parser

    parser = argparse.ArgumentParser(description='TF2 classification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = add_arguments(parser)
    args = parser.parse_args()
    logging.info(f'args = {args}')

    dataset_name = args.dataset
    dataset_path = args.dataset_path

    ds_train, _, ds_size, _, _ = DatasetFactory.get_dataset(
        dataset_name=dataset_name, dataset_path=dataset_path, split='train', img_datatype=float,
        micro_batch_size=1, accelerator_side_preprocess=False, apply_preprocessing=False)

    train_split_match = DataGenerator.evaluate_size_dataset(ds_train) == ds_size

    ds_valid, _, ds_valid_size, _, _ = DatasetFactory.get_dataset(
        dataset_name=dataset_name, dataset_path=dataset_path, split='test', img_datatype=float,
        micro_batch_size=1, accelerator_side_preprocess=False, apply_preprocessing=False)

    test_split_match = DataGenerator.evaluate_size_dataset(ds_valid) == ds_valid_size

    if not train_split_match:
        logging.warning(
            f'Train split of "{dataset_name}" provided at "{dataset_path}" does not match the size DatasetFactory returned.')

    if not test_split_match:
        logging.warning(
            f'Test split of dataset "{dataset_name}" provided at "{dataset_path}" does not match the size DatasetFactory returned.')

    if train_split_match and test_split_match:
        logging.info(f'Dataset "{dataset_name}" provided at "{dataset_path}" matches the size DatasetFactory returned.')
