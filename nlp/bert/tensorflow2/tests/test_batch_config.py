# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest

from data_utils.batch_config import BatchConfig, Task


class TestBatchConfig:
    def test_micro_batch_size(self):
        batch_config = BatchConfig(micro_batch_size=3, num_replicas=1, gradient_accumulation_count=1, dataset_size=18)
        assert batch_config.micro_batch_size == 3

    def test_num_replicas(self):
        batch_config = BatchConfig(micro_batch_size=1, num_replicas=4, gradient_accumulation_count=1, dataset_size=40)
        assert batch_config.num_replicas == 4

    def test_num_micro_batches_per_weight_update(self):
        batch_config = BatchConfig(micro_batch_size=1, num_replicas=4, gradient_accumulation_count=4, dataset_size=40)
        assert batch_config.num_micro_batches_per_weight_update == 4 * 4

    def test_gradient_accumulation_count(self):
        batch_config = BatchConfig(micro_batch_size=1, num_replicas=1, gradient_accumulation_count=2, dataset_size=40)
        assert batch_config.gradient_accumulation_count == 2

    def test_calc_global_batch_size(self):
        batch_config = BatchConfig(micro_batch_size=3, num_replicas=4, gradient_accumulation_count=2, dataset_size=40)
        assert batch_config.global_batch_size == 24

    def test_get_num_micro_batches_per_epoch(self):
        batch_config = BatchConfig(micro_batch_size=3, num_replicas=2, gradient_accumulation_count=3, dataset_size=18)
        assert batch_config.num_micro_batches_per_epoch == 6

        batch_config = BatchConfig(micro_batch_size=3, num_replicas=2, gradient_accumulation_count=3, dataset_size=20)
        assert batch_config.num_micro_batches_per_epoch == 6

    def test_num_train_steps(self):
        batch_config = BatchConfig(micro_batch_size=2, num_replicas=2, gradient_accumulation_count=2, dataset_size=16)
        assert batch_config.num_train_steps == 57606144
        assert batch_config.epochs == 28803072
        assert batch_config.num_micro_batches_per_epoch == 8
        assert batch_config.total_num_micro_batches == 230424576

    def test_custom_num_train_steps(self):
        batch_config = BatchConfig(
            micro_batch_size=2,
            num_replicas=2,
            gradient_accumulation_count=2,
            dataset_size=8,
            total_num_train_samples=80,
        )
        assert batch_config.num_train_steps == 10
        assert batch_config.epochs == 10
        assert batch_config.total_num_micro_batches == 40

    def test_total_num_batches(self):
        batch_config = BatchConfig(
            micro_batch_size=2, num_replicas=2, gradient_accumulation_count=2, dataset_size=16, global_batches_per_log=1
        )
        assert batch_config.num_micro_batches_per_epoch == 8
        assert batch_config.steps_per_execution == 2
        assert batch_config.epochs == 28803072
        assert batch_config.total_num_micro_batches == 230424576

        batch_config = BatchConfig(
            micro_batch_size=2, num_replicas=2, gradient_accumulation_count=2, dataset_size=16, global_batches_per_log=2
        )
        assert batch_config.num_micro_batches_per_epoch == 8
        assert batch_config.steps_per_execution == 4
        assert batch_config.epochs == 28803072
        assert batch_config.total_num_micro_batches == 230424576

        batch_config = BatchConfig(
            micro_batch_size=2, num_replicas=2, gradient_accumulation_count=2, dataset_size=18, global_batches_per_log=2
        )
        assert batch_config.num_micro_batches_per_epoch == 8
        assert batch_config.steps_per_execution == 4
        assert batch_config.total_num_micro_batches == 204821840

    def test_raise_error_for_no_phase(self):
        with pytest.raises(ValueError):
            BatchConfig(
                micro_batch_size=2, num_replicas=2, gradient_accumulation_count=2, dataset_size=40, task=Task.OTHER
            )

    def test_raise_error_for_incompatible_pipeline_grad_acc(self):
        # Num pipeline stages of 1 is allowed
        BatchConfig(
            micro_batch_size=1, num_replicas=1, gradient_accumulation_count=2, num_pipeline_stages=1, dataset_size=40
        )
        # Num pipeline stages greater than 1 is not allowed
        with pytest.raises(ValueError):
            BatchConfig(
                micro_batch_size=1,
                num_replicas=1,
                gradient_accumulation_count=2,
                num_pipeline_stages=2,
                dataset_size=40,
            )
