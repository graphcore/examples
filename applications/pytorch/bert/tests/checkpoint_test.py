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
from modeling import PipelinedBertForPretraining
from utils import parse_bert_args
from optimization import get_optimizer
from checkpointing import restore_checkpoint, save_checkpoint, checkpoints_exist, _get_checkpoint_filename
import tempfile


@pytest.mark.category1
@pytest.mark.parametrize("embedding_serialization", (1, 5))
@pytest.mark.parametrize("recompute_checkpoint", (False, True))
def test_checkpoint_save_restore(recompute_checkpoint, embedding_serialization):
    """
    Test that saving and restoring checkpoints works. Also test checkpointing
    with recomputation checkpoints and embedding serialization.
    """
    args = """
    --config unit_test
    """.split()
    config = BertConfig(**(vars(parse_bert_args(args))))
    config.recompute_checkpoint_every_layer = recompute_checkpoint
    config.embedding_serialization = embedding_serialization
    model1 = PipelinedBertForPretraining(config)
    model2 = PipelinedBertForPretraining(config)
    optimizer = get_optimizer(config, model1)

    # The two models should have different initial weights
    for name, tensor1 in model1.state_dict().items():
        tensor2 = model2.state_dict()[name]
        if (tensor1.dtype is not torch.int64) and ("LayerNorm" not in name) and ("bias" not in name):
            assert not torch.allclose(tensor1, tensor2)

    # Save and restore checkpoint
    with tempfile.TemporaryDirectory() as dir:
        config.checkpoint_dir = dir
        # No checkpoints should exist yet
        assert not checkpoints_exist(config)

        save_checkpoint(config, model1, optimizer, 0)

        # Checkpoint should now exist
        assert checkpoints_exist(config)

        # Restore from checkpoint
        config.checkpoint_file = os.path.join(dir, _get_checkpoint_filename(config, 0))
        checkpoint = restore_checkpoint(config)
        model2.load_state_dict(checkpoint["model_state_dict"])

        # Models should now have the same weights
        for name, tensor1 in model1.state_dict().items():
            tensor2 = model2.state_dict()[name]
            assert torch.allclose(tensor1, tensor2)


@pytest.mark.category1
@pytest.mark.parametrize("embedding_serialization", (1, 5))
def test_checkpoint_embedding_serialization(embedding_serialization):
    """
    If a checkpoint is saved with embedding_serialization
      then we should be able to restore the checkpoint in a new run
      where embedding_serialization isn't used.
    The reverse should also hold.
    """
    args = """
    --config unit_test
    """.split()
    config1 = BertConfig(**(vars(parse_bert_args(args))))
    config1.embedding_serialization = embedding_serialization
    model1 = PipelinedBertForPretraining(config1)

    with tempfile.TemporaryDirectory() as dir:
        # Save checkpoint
        config1.checkpoint_dir = dir
        save_checkpoint(config1, model1, get_optimizer(config1, model1), 0)

        # New model with opposite embedding_serialization to model1
        config2 = BertConfig(**(vars(parse_bert_args(args))))
        config2.embedding_serialization = 5 if embedding_serialization == 1 else 1
        model2 = PipelinedBertForPretraining(config2)

        # Restore
        config2.checkpoint_file = os.path.join(dir, _get_checkpoint_filename(config1, 0))
        checkpoint = restore_checkpoint(config2)
        model2.load_state_dict(checkpoint["model_state_dict"])

        # Models should now have the same weights
        for name, tensor1 in model1.state_dict().items():
            tensor2 = model2.state_dict()[name]
            assert torch.allclose(tensor1, tensor2)


@pytest.mark.category1
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
    model1 = PipelinedBertForPretraining(config1)
    optimizer = get_optimizer(config1, model1)

    with tempfile.TemporaryDirectory() as dir:
        # Save checkpoint
        config1.checkpoint_dir = dir
        save_checkpoint(config1, model1, get_optimizer(config1, model1), 0)

        # New model with opposite `recompute_checkpoint` to model1
        config2 = BertConfig(**(vars(parse_bert_args(args))))
        config2.recompute_checkpoint_every_layer = not recompute_checkpoint
        model2 = PipelinedBertForPretraining(config2)

        # Restore
        config2.checkpoint_file = os.path.join(dir, _get_checkpoint_filename(config1, 0))
        checkpoint = restore_checkpoint(config2)
        model2.load_state_dict(checkpoint["model_state_dict"])

        # Models should now have the same weights
        for name, tensor1 in model1.state_dict().items():
            tensor2 = model2.state_dict()[name]
            assert torch.allclose(tensor1, tensor2)


@pytest.mark.category1
def test_checkpoint_incompatible():
    """
    Checkpoints with different hidden size, num attention heads, and number of layers
    should be incompatible.
    """
    def _test_incompatible(config1, config2):
        model1 = PipelinedBertForPretraining(config1)
        with tempfile.TemporaryDirectory() as dir:
            # Save checkpoint
            config1.checkpoint_dir = dir
            save_checkpoint(config1, model1, get_optimizer(config1, model1), 0)

            # Restore should throw exception
            config2.checkpoint_file = os.path.join(dir, _get_checkpoint_filename(config1, 0))
            with pytest.raises(RuntimeError) as excinfo:
                _ = restore_checkpoint(config2)
            assert "Checkpoint being loaded does not match model definition" in str(excinfo)

    # Different hidden size shouldn't work
    args1 = """
    --config unit_test
    --hidden-size 32
    """.split()
    args2 = """
    --config unit_test
    --hidden-size 64
    """.split()
    config1 = BertConfig(**(vars(parse_bert_args(args1))))
    config2 = BertConfig(**(vars(parse_bert_args(args2))))
    _test_incompatible(config1, config2)

    # Different number of layers shouldn't work
    args1 = """
    --config unit_test
    --num-hidden-layers 12
    --layers-per-ipu 12
    """.split()
    args2 = """
    --config unit_test
    --num-hidden-layers 24
    --layers-per-ipu 24
    """.split()
    config1 = BertConfig(**(vars(parse_bert_args(args1))))
    config2 = BertConfig(**(vars(parse_bert_args(args2))))
    _test_incompatible(config1, config2)

    # Different number of attention heads shouldn't work
    args1 = """
    --config unit_test
    --num-attention-heads 16
    """.split()
    args2 = """
    --config unit_test
    --num-attention-heads 8
    """.split()
    config1 = BertConfig(**(vars(parse_bert_args(args1))))
    config2 = BertConfig(**(vars(parse_bert_args(args2))))
    _test_incompatible(config1, config2)
