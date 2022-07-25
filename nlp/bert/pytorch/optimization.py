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

from transformers import get_linear_schedule_with_warmup
from transformers import get_constant_schedule
from poptorch.optim import AdamW, LAMB
from torch import float16, float32
import horovod.torch as hvd


def get_lr_scheduler(optimizer,
                     scheduler_type,
                     lr_warmup=None,
                     num_steps=None):
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(lr_warmup * num_steps), num_steps)
    elif scheduler_type == "constant":
        scheduler = get_constant_schedule(optimizer)
    else:
        raise ValueError("Unknown scheduler_type:", scheduler_type)

    # Initialize step as Poptorch does not call optimizer.step() explicitly
    optimizer._step_count = 1

    return scheduler


def get_optimizer(config, model):

    # Do not apply weight_decay for one-dimensional parameters
    # Additionally, do not apply LAMB update for biases
    params = [{"params": [], "weight_decay": 0, "max_weight_norm": 0},
              {"params": [], "weight_decay": 0},
              {"params": [], "weight_decay": config.weight_decay}]
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bias" in name:
                params[0]["params"].append(param)
            elif len(param.shape) == 1:
                params[1]["params"].append(param)
            else:
                params[2]["params"].append(param)

    first_order_type = float16 if config.enable_half_first_order_momentum else float32

    if config.optimizer == "AdamW":
        optimizer = AdamW(params,
                          lr=config.learning_rate,
                          weight_decay=0,
                          eps=1e-6,
                          bias_correction=False,
                          loss_scaling=config.loss_scaling,
                          accum_type=float16,
                          first_order_momentum_accum_type=first_order_type,
                          second_order_momentum_accum_type=float32)
    elif config.optimizer in ("LAMBNoBiasCorrection", "LAMB"):
        optimizer = LAMB(params,
                         lr=config.learning_rate,
                         weight_decay=0,
                         eps=1e-6,
                         loss_scaling=config.loss_scaling,
                         max_weight_norm=None,
                         accum_type=float16,
                         first_order_momentum_accum_type=first_order_type,
                         second_order_momentum_accum_type=float32,
                         bias_correction=config.optimizer == "LAMB")
        optimizer.variable_attrs.markAsConstant("max_weight_norm")
    else:
        raise ValueError("Unknown Optimizer:", config.optimizer)

    optimizer.variable_attrs.markAsConstant("weight_decay")

    # Make optimizers distributed
    if config.use_popdist:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    return optimizer
