# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import pytest
import os
from transformers import BertConfig
from modeling import PipelinedBertForPretraining, PipelinedBertForQuestionAnswering
from args import parse_bert_args
from checkpointing import save_checkpoint, checkpoints_exist
import tempfile


@pytest.mark.parametrize("embedding_serialization_factor", (1, 5))
@pytest.mark.parametrize("recompute_checkpoint", (False, True))
def test_checkpoint_save_restore(recompute_checkpoint, embedding_serialization_factor):
    """
    Test that saving and restoring checkpoints works. Also test checkpointing
    with recomputation checkpoints and embedding serialization.
    """
    args = """
    --config unit_test
    """.split()
    config = BertConfig(**(vars(parse_bert_args(args))))
    config.recompute_checkpoint_every_layer = recompute_checkpoint
    config.embedding_serialization_factor = embedding_serialization_factor
    model1 = PipelinedBertForPretraining(config).parallelize()
    model2 = PipelinedBertForPretraining(config).parallelize()

    # The two models should have different initial weights
    for name, tensor1 in model1.state_dict().items():
        tensor2 = model2.state_dict()[name]
        if (tensor1.dtype is not torch.int64) and ("LayerNorm" not in name) and ("bias" not in name):
            assert not torch.allclose(tensor1, tensor2)

    # Save and restore checkpoint
    with tempfile.TemporaryDirectory() as dir:
        config.checkpoint_output_dir = dir
        # No checkpoints should exist yet
        assert not checkpoints_exist(config.checkpoint_output_dir)

        save_checkpoint(config, model1, 0)

        # Checkpoint should now exist
        assert checkpoints_exist(config.checkpoint_output_dir)

        # Restore from checkpoint
        model2 = PipelinedBertForPretraining.from_pretrained(os.path.join(dir, "step_0"), config=config)

        # Models should now have the same weights
        for name, tensor1 in model1.state_dict().items():
            tensor2 = model2.state_dict()[name]
            assert torch.allclose(tensor1, tensor2)


@pytest.mark.parametrize("embedding_serialization_factor", (1, 5))
def test_checkpoint_embedding_serialization(embedding_serialization_factor):
    """
    If a checkpoint is saved with embedding_serialization_factor
      then we should be able to restore the checkpoint in a new run
      where embedding_serialization_factor isn't used.
    The reverse should also hold.
    """
    args = """
    --config unit_test
    """.split()
    config1 = BertConfig(**(vars(parse_bert_args(args))))
    config1.embedding_serialization_factor = embedding_serialization_factor
    model1 = PipelinedBertForPretraining(config1).parallelize()

    with tempfile.TemporaryDirectory() as dir:
        # Save checkpoint
        config1.checkpoint_output_dir = dir
        save_checkpoint(config1, model1, 0)

        # New model with opposite embedding_serialization to model1
        config2 = BertConfig(**(vars(parse_bert_args(args))))
        config2.embedding_serialization_factor = 5 if embedding_serialization_factor == 1 else 1
        model2 = PipelinedBertForPretraining.from_pretrained(os.path.join(dir, "step_0"), config=config2).parallelize()

        assert model2.config.embedding_serialization_factor == config2.embedding_serialization_factor

        # Models should now have the same weights
        for name, tensor1 in model1.state_dict().items():
            tensor2 = model2.state_dict()[name]
            assert torch.allclose(tensor1, tensor2)


@pytest.mark.parametrize("embedding_serialization_factor", (1, 5))
def test_checkpoint_embedding_serialization_qa(embedding_serialization_factor):
    """
    If a checkpoint is saved with embedding_serialization_factor
      then we should be able to restore the checkpoint in a new run
      where embedding_serialization_factor isn't used.
    The reverse should also hold.
    For PipelinedBertForQuestionAnswering we will need to call `deparallelize`
    before checkpointing.
    """
    args = """
    --config unit_test
    """.split()
    config = BertConfig(**(vars(parse_bert_args(args))))
    config.embedding_serialization_factor = embedding_serialization_factor
    model1 = PipelinedBertForQuestionAnswering(config).parallelize()

    with tempfile.TemporaryDirectory() as dir:
        # Save checkpoint
        config.checkpoint_output_dir = dir
        model1.deparallelize()
        save_checkpoint(config, model1, 0)

        # Load the checkpoint, but don't call parallelize
        model2 = PipelinedBertForQuestionAnswering.from_pretrained(os.path.join(dir, "step_0"))

        # Models should have the same weights
        for name, tensor1 in model1.state_dict().items():
            tensor2 = model2.state_dict()[name]
            assert torch.allclose(tensor1, tensor2)


@pytest.mark.parametrize("recompute_checkpoint", (True, False))
def test_checkpoint_recompute_checkpoint(recompute_checkpoint):
    """
    If a checkpoint is saved with `recompute_checkpoint_every_layer`
      then we should be able to restore the checkpoint in a new run
      that doesn't use `recompute_checkpoint_every_layer` and vice-verse.
    """
    args = """
    --config unit_test
    """.split()
    config1 = BertConfig(**(vars(parse_bert_args(args))))
    config1.recompute_checkpoint_every_layer = recompute_checkpoint
    model1 = PipelinedBertForPretraining(config1).parallelize()

    with tempfile.TemporaryDirectory() as dir:
        # Save checkpoint
        config1.checkpoint_output_dir = dir
        save_checkpoint(config1, model1, 0)

        # New model with opposite `recompute_checkpoint` to model1
        config2 = BertConfig(**(vars(parse_bert_args(args))))
        config2.recompute_checkpoint_every_layer = not recompute_checkpoint
        model2 = PipelinedBertForPretraining.from_pretrained(os.path.join(dir, "step_0"), config=config2).parallelize()

        # Models should now have the same weights
        for name, tensor1 in model1.state_dict().items():
            tensor2 = model2.state_dict()[name]
            assert torch.allclose(tensor1, tensor2)
