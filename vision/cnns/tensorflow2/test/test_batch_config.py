# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import sys
import unittest

sys.path.append(str(Path(__file__).absolute().parent.parent))
from batch_config import BatchConfig


class BatchConfigTest(unittest.TestCase):
    def test_micro_batch_size(self):
        batch_config = BatchConfig(micro_batch_size=3, num_replicas=1, gradient_accumulation_count=1)
        assert batch_config.micro_batch_size == 3

    def test_num_replicas(self):
        batch_config = BatchConfig(micro_batch_size=1, num_replicas=4, gradient_accumulation_count=1)
        assert batch_config.num_replicas == 4

    def test_num_micro_batches_per_weight_update(self):
        batch_config = BatchConfig(micro_batch_size=1, num_replicas=4, gradient_accumulation_count=4)
        assert batch_config.num_micro_batches_per_weight_update == 4 * 4

    def test_gradient_accumulation_count(self):
        batch_config = BatchConfig(micro_batch_size=1, num_replicas=1, gradient_accumulation_count=2)
        assert batch_config.gradient_accumulation_count == 2

    def test_calc_global_batch_size(self):
        batch_config = BatchConfig(micro_batch_size=3, num_replicas=4, gradient_accumulation_count=2)
        assert batch_config.global_batch_size == 24

    def test_global_batch_size(self):
        batch_config = BatchConfig(micro_batch_size=1, num_replicas=1, global_batch_size=5)
        assert batch_config.global_batch_size == 5

    def test_calc_gradient_accumulation_count(self):
        batch_config = BatchConfig(micro_batch_size=3, num_replicas=2, global_batch_size=12)
        assert batch_config.gradient_accumulation_count == 2

    def test_round_gradient_accumulation_count(self):
        batch_config = BatchConfig(micro_batch_size=3, num_replicas=2, global_batch_size=10)
        assert batch_config.gradient_accumulation_count == 2
        assert batch_config.global_batch_size == 12

    def test_get_num_micro_batches_per_epoch(self):
        batch_config = BatchConfig(micro_batch_size=3, num_replicas=2, global_batch_size=10)
        assert batch_config.get_num_micro_batches_per_epoch(dataset_size=18) == 4
        assert batch_config.get_num_micro_batches_per_epoch(dataset_size=20) == 4

    def test_exclusive_global_batch_size_gradient_accumulation_count(self):
        self.assertRaises(
            ValueError,
            BatchConfig,
            micro_batch_size=3,
            num_replicas=2,
            gradient_accumulation_count=1,
            global_batch_size=6,
        )

    def test_missing_gradient_accumulation_count_global_batch_size(self):
        self.assertRaises(ValueError, BatchConfig, micro_batch_size=1, num_replicas=1)
