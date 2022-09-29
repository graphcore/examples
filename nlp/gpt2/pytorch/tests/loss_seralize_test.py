# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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


import pytest
import numpy as np
import poptorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from poptorch import trainingModel
from poptorch.optim import AdamW
from torch import float16, float32


Seralize_block_size = 128
hidden_size = 256
vocab_size = 30000
input_len = 256
device_iterations = 1
bs = 1
replica = 1
gradient_accumulation = 6
random_seed = 1472
target_num = input_len
optimizer_state_offchip = True
replicated_tensor_sharding = False
mem_prop = {"IPU0": 0.1}


class SerializedLinear(nn.Linear):
    def __init__(self, in_features, out_features, factor, bias=False,
                 mode=poptorch.MatMulSerializationMode.OutputChannels):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.factor = factor

    def forward(self, x):
        size_out = x.size()[:-1] + (self.out_features,)
        output = poptorch.serializedMatMul(
            x, self.weight.t(), self.mode, self.factor)
        if self.bias is not None:
            output += self.bias
        return output.view(*size_out)


class Model(nn.Module):
    def __init__(self, serialize=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.matmul = SerializedLinear(hidden_size, vocab_size, 4)
        self.matmul.weight = self.emb.weight
        self.emb = poptorch.BeginBlock(self.emb, ipu_id=0)
        self.linear = poptorch.BeginBlock(self.linear, ipu_id=1)
        self.matmul = poptorch.BeginBlock(self.matmul, ipu_id=0)
        self.serialize = serialize

    def forward(self, input_ids, labels):
        hidden_states = self.emb(input_ids)
        hidden_states = self.linear(hidden_states)
        hidden_states = poptorch.recomputationCheckpoint(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        logits = self.matmul(hidden_states)
        if not self.serialize:
            loss_weights = torch.sum(
                (labels.view(-1) > -1).to(torch.float), dim=-1)
            self.loss_fct = nn.CrossEntropyLoss(reduction="sum")
            logits = self.matmul(hidden_states)
            loss = self.loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1)).to(torch.float32)
            mean_loss = loss / loss_weights
            total_loss = poptorch.identity_loss(mean_loss, reduction="none")
            return total_loss
        else:
            self.loss_fct = nn.CrossEntropyLoss(reduction="sum")
            labels = labels.view(-1)
            loss = None
            loss_weights = torch.sum(
                (labels > -1).to(torch.float), dim=-1).to(torch.float32)
            for i in range(Seralize_block_size, input_len+Seralize_block_size, Seralize_block_size):
                logit = logits[i-Seralize_block_size:i, :]
                label = labels[i-Seralize_block_size:i]
                if loss is None:
                    loss = self.loss_fct(logit, label).to(torch.float32)
                    loss = poptorch.recomputationCheckpoint(loss)
                else:
                    tmp_loss = self.loss_fct(logit, label).to(torch.float32)
                    tmp_loss = poptorch.recomputationCheckpoint(tmp_loss)
                    loss = loss + tmp_loss
            loss /= loss_weights
            total_loss = poptorch.identity_loss(loss, reduction="none")
            return total_loss


@pytest.mark.ipus(1)
def test_loss_seralize():
    """
    Test that the loss split operation in the LM layer.
    It shuold give the same result with normal LM layer.
    """
    opts = poptorch.Options()
    opts.Precision.enableStochasticRounding(False)
    opts.outputMode(poptorch.OutputMode.Sum)
    opts.replicationFactor(replica)
    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(device_iterations)
    opts.Training.gradientAccumulation(gradient_accumulation)
    opts.Training.accumulationAndReplicationReductionType(
        poptorch.ReductionType.Mean)
    np.random.seed(random_seed)
    opts.randomSeed(random_seed)
    opts.setAvailableMemoryProportion(mem_prop)
    # Use Pipelined Execution
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))
    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings()
        .useOnChipStorage(not optimizer_state_offchip)
        .useReplicatedTensorSharding(replicated_tensor_sharding))
    opts.Precision.setPartialsType(torch.float16)
    opts._Popart.set("decomposeGradSum", True)
    opts._Popart.set("scheduleNonWeightUpdateGradientConsumersEarly", True)
    opts._Popart.setPatterns(
        {"TiedGather": True, "TiedGatherAccumulate": True, "UpdateInplacePrioritiesForIpu": True})

    model0 = Model(serialize=False).half().train()
    model1 = Model(serialize=True).half().train()
    model0.load_state_dict(model1.state_dict())

    params0 = [{"params": [], "weight_decay": 0, "max_weight_norm": 0},
               {"params": [], "weight_decay": 0},
               {"params": [], "weight_decay": 0.1}]
    params1 = [{"params": [], "weight_decay": 0, "max_weight_norm": 0},
               {"params": [], "weight_decay": 0},
               {"params": [], "weight_decay": 0.1}]
    for name, param in model0.named_parameters():
        if param.requires_grad:
            if "bias" in name:
                params0[0]["params"].append(param)
            elif len(param.shape) == 1:
                params0[1]["params"].append(param)
            else:
                params0[2]["params"].append(param)
    for name, param in model1.named_parameters():
        if param.requires_grad:
            if "bias" in name:
                params1[0]["params"].append(param)
            elif len(param.shape) == 1:
                params1[1]["params"].append(param)
            else:
                params1[2]["params"].append(param)
    optimizer0 = AdamW(params0,
                       lr=1e-5,
                       weight_decay=0,
                       eps=1e-6,
                       bias_correction=False,
                       loss_scaling=1,
                       accum_type=float16,
                       first_order_momentum_accum_type=float16,
                       second_order_momentum_accum_type=float32)
    optimizer1 = AdamW(params1,
                       lr=1e-5,
                       weight_decay=0,
                       eps=1e-6,
                       bias_correction=False,
                       loss_scaling=1,
                       accum_type=float16,
                       first_order_momentum_accum_type=float16,
                       second_order_momentum_accum_type=float32)

    poptorch_model0 = trainingModel(model0, opts, optimizer=optimizer0)
    poptorch_model1 = trainingModel(model1, opts, optimizer=optimizer1)

    inputs = []
    for _ in range(3):
        input1 = torch.randint(0, 20000, [device_iterations*bs*replica *
                                          gradient_accumulation, input_len], dtype=torch.long)
        input2 = torch.randint(0, 2, [device_iterations*bs*replica *
                                      gradient_accumulation, target_num], dtype=torch.long)
        inputs.append((input1, input2))

    output0 = []
    for (input1, input2) in inputs:
        outputs = poptorch_model0(input1, input2)
        poptorch_model0.setOptimizer(optimizer0)
        output0.append(outputs.numpy())

    output1 = []
    for (input1, input2) in inputs:
        outputs = poptorch_model1(input1, input2)
        poptorch_model1.setOptimizer(optimizer1)
        output1.append(outputs.numpy())

    assert np.allclose(output0, output1, rtol=1e-3)
