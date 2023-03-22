# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import argparse
import logging
import tensorflow as tf

from datasets.dataset_factory import DatasetFactory
from batch_config import BatchConfig


def main(dataset, dataset_path):

    batch_config = BatchConfig(micro_batch_size=1, num_replicas=1, gradient_accumulation_count=1)

    app_train_dataset, _, _ = DatasetFactory.get_dataset(
        dataset_name=dataset,
        dataset_path=dataset_path,
        split="train",
        img_datatype=tf.float32,
        batch_config=batch_config,
        accelerator_side_preprocess=False,
        repeat=False,
    )

    evaluated_train_size = app_train_dataset.evaluate_size(batch_config.micro_batch_size)
    train_split_match = evaluated_train_size == app_train_dataset.size

    app_test_dataset, _, _ = DatasetFactory.get_dataset(
        dataset_name=dataset,
        dataset_path=dataset_path,
        split="test",
        img_datatype=tf.float32,
        batch_config=batch_config,
        accelerator_side_preprocess=False,
        repeat=False,
    )

    evaluated_test_size = app_test_dataset.evaluate_size(batch_config.micro_batch_size)
    test_split_match = evaluated_test_size == app_test_dataset.size

    if not train_split_match:
        logging.warning(
            f'Train split of "{dataset}" provided at "{dataset_path}" does not match expected size. '
            f"Expected {app_train_dataset.size}, found {evaluated_train_size}"
        )

    if not test_split_match:
        logging.warning(
            f'Test split of dataset "{dataset}" provided at "{dataset_path}" does not match expected size. '
            f"Expected {app_test_dataset.size}, found {evaluated_test_size}"
        )

    if train_split_match and test_split_match:
        logging.info(f'Dataset "{dataset}" provided at "{dataset_path}" matches the expected size.')


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # add arg parser
    def add_arguments(parser):
        parser.add_argument("--dataset", type=str, default="cifar10", help="Name of dataset to use")
        parser.add_argument("--dataset-path", type=str, default=".", help="Path to dataset")
        return parser

    parser = argparse.ArgumentParser(
        description="TF2 classification", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser = add_arguments(parser)
    args = parser.parse_args()
    logging.info(f"args = {args}")
    main(**vars(args))
