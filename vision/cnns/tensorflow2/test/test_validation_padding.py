# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import unittest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))
from batch_config import BatchConfig

IMAGENET_VALIDATION_SIZE = 50000


class ValidationPaddingTest(unittest.TestCase):

    def with_parameters(self,
                        micro_batch_size,
                        num_replicas,
                        num_instances,
                        expected_discarded_samples,
                        expected_padding_samples,
                        expected_padded_dataset_size):

        batch_config = BatchConfig(micro_batch_size=micro_batch_size,
                                   num_replicas=num_replicas,
                                   gradient_accumulation_count=1)
        discarded_samples = batch_config.get_num_discarded_samples_per_instance(dataset_size=IMAGENET_VALIDATION_SIZE,
                                                                                num_instances=num_instances)
        self.assertEqual(discarded_samples, expected_discarded_samples)

        padding_samples = batch_config.get_num_padding_samples_per_instance(num_discarded_samples_per_instance=discarded_samples,
                                                                            num_instances=num_instances)
        self.assertEquals(padding_samples, expected_padding_samples)

        padded_dataset_size = batch_config.get_padded_dataset_size(IMAGENET_VALIDATION_SIZE, num_instances=num_instances)
        self.assertEquals(padded_dataset_size, expected_padded_dataset_size)


    def test_mlperf_bs16_pod16_1instance(self):
        self.with_parameters(micro_batch_size=16,
                             num_replicas=16,
                             num_instances=1,
                             expected_discarded_samples=80,
                             expected_padding_samples=176,
                             expected_padded_dataset_size=50176)

    def test_mlperf_bs16_pod16_8instances(self):
        self.with_parameters(micro_batch_size=16,
                             num_replicas=16,
                             num_instances=8,
                             expected_discarded_samples=10,
                             expected_padding_samples=22,
                             expected_padded_dataset_size=50176)

    def test_mlperf_bs16_pod64_1instance(self):
        self.with_parameters(micro_batch_size=16,
                             num_replicas=64,
                             num_instances=1,
                             expected_discarded_samples=848,
                             expected_padding_samples=176,
                             expected_padded_dataset_size=50176)

    def test_mlperf_bs16_pod64_32instances(self):
        self.with_parameters(micro_batch_size=16,
                             num_replicas=64,
                             num_instances=32,
                             expected_discarded_samples=26,
                             expected_padding_samples=6,
                             expected_padded_dataset_size=50176)
