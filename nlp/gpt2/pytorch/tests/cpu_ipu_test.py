# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings

import pytest
import torch
import poptorch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2Config, GPT2LMHeadModel

import import_helper
from arguments import set_args
from train_gpt2 import GPT2Wrapper
from model.optimized_gpt2_attn import OptimizedGPT2Attention

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

base_dir = os.path.abspath(os.path.dirname(__file__))


class cpu_wrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = GPT2LMHeadModel(config=config)
        for layer in self.model.transformer.h:
            gpt2_attn = OptimizedGPT2Attention(
                self.model.config, layer_idx=layer.attn.layer_idx)
            gpt2_attn.load_state_dict(layer.attn.state_dict())
            layer.attn = gpt2_attn

    def forward(self, input_ids, labels):
        transformer_outputs = self.model.transformer(input_ids=input_ids)
        hidden_states = transformer_outputs[0]
        lm_logits = self.model.lm_head(hidden_states)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        loss = poptorch.identity_loss(loss, reduction="none")
        acc = torch.Tensor(0)
        return loss, acc


@pytest.mark.ipus(1)
def test_ipu_cpu_match():
    """
    Test that the GPT2 model ran on IPU approximately matches that same
    model ran on the CPU.
    """

    # Config
    args = set_args()
    args.batch_size = 1
    args.pretrained_model = None
    args.embedding_serialization_factor = 2
    args.layers_per_ipu = [1, 3]
    args.matmul_proportion = [0.2, 0.2]
    args.recompute_checkpoint_every_layer = True

    batch_size = args.batch_size
    config = GPT2Config.from_json_file(base_dir + '/../config/config.json')
    config.model = 'gpt2'
    config.attn_pdrop = 0.0
    config.embd_pdrop = 0.0
    config.resid_pdrop = 0.0
    config.summary_first_dropout = 0.0
    config.activation_function = "gelu"
    config.n_layer = 4
    config.n_embd = 256
    config.n_head = 2
    config.vocab_size = 20256
    config.n_positions = 128

    # Models and options
    opts = poptorch.Options().deviceIterations(1)
    opts.setExecutionStrategy(poptorch.ShardedExecution(
        poptorch.AutoStage.AutoIncrement))
    opts.Training.gradientAccumulation(1)
    opts.replicationFactor(1)
    opts.Precision.setPartialsType(torch.float32)
    opts.outputMode(poptorch.OutputMode.Final)
    opts.randomSeed(1234)

    model_cpu = cpu_wrapper(config=config).train()
    model_ipu = GPT2Wrapper(args, config).train()
    model_ipu.load_state_dict(model_cpu.state_dict())

    # Check that copy was successful
    assert model_ipu is not model_cpu
    assert all([(a == b).all() for a, b in zip(
        model_cpu.parameters(), model_ipu.parameters())]) is True

    optimizer_cpu = torch.optim.AdamW(model_cpu.parameters(), lr=0.001)
    optimizer_ipu = poptorch.optim.AdamW(
        model_ipu.model.parameters(), lr=0.001, loss_scaling=1.0)
    poptorch_model = poptorch.trainingModel(
        model_ipu, opts, optimizer=optimizer_ipu)

    # Input
    tokens = torch.randint(0, 20256, (129, ))
    labels = torch.tensor(tokens[1:])
    tokens = torch.tensor(tokens[:-1])
    batch_input = (tokens.repeat(batch_size, 1), labels.repeat(batch_size, 1))

    # Training Loop
    for step in range(10):
        # Step CPU model
        optimizer_cpu.zero_grad()
        cpu_output = model_cpu(*batch_input)
        cpu_loss = cpu_output[0]
        cpu_loss.backward()
        optimizer_cpu.step()

        # Step IPU Model
        ipu_output = poptorch_model(*batch_input)
        ipu_loss = ipu_output[0]

        with torch.no_grad():
            print(f"CPU Loss: {cpu_loss}, IPU Loss: {ipu_loss}")
            # Check the losses are approximately equal
            assert np.allclose(cpu_loss.numpy(), ipu_loss.numpy(), rtol=1e-3)
