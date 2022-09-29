# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))

from datasets import dataset_factory
from batch_config import BatchConfig

if __name__ == '__main__':

    batch_config = BatchConfig(micro_batch_size=1,
                               num_replicas=1,
                               gradient_accumulation_count=1)

    splits = ['train', 'test']

    # instantiate cifar10 to download it
    for split in splits:
        cifar = dataset_factory.DatasetFactory.get_dataset(
            dataset_name='cifar10',
            dataset_path='.',
            split=split,
            img_datatype=tf.float32,
            batch_config=batch_config
        )