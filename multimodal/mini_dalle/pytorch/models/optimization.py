# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from poptorch.optim import AdamW, Adam
from torch import float16, float32
import horovod.torch as hvd


def warmup_multi_step(step, warmup_step, total_step):
    # Multi step schedule with warmup strategy
    if step < warmup_step:
        return (1 + step) / (1 + warmup_step)
    elif (step - warmup_step) < (total_step - warmup_step) / 2:
        return 1
    else:
        return 1e-1


def get_lr_sched(global_step, scheduler, num_train_steps, warmup_ratio=0.2):
    warmup_steps = int(warmup_ratio * num_train_steps)
    if scheduler == "multi_step":
        lr_ratio = warmup_multi_step(global_step, warmup_steps, num_train_steps)
    elif scheduler == "constant":
        if global_step < warmup_steps:
            lr_ratio = (1 + global_step) / (1 + warmup_steps)
        else:
            lr_ratio = 1.0
    return lr_ratio


def get_optimizer(config, model):
    # Do not apply weight_decay for one-dimensional parameters
    regularized_params = []
    non_regularized_params = []
    for param in model.parameters():
        if param.requires_grad:
            if len(param.shape) == 1:
                non_regularized_params.append(param)
            else:
                regularized_params.append(param)

    params = [
        {"params": regularized_params, "weight_decay": config.weight_decay},
        {"params": non_regularized_params, "weight_decay": 0},
    ]

    first_order_type = float16 if config.enable_half_first_order_momentum else float32

    if config.optimizer == "AdamW":
        optimizer = AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=0,
            eps=1e-6,
            bias_correction=False,
            loss_scaling=config.loss_scaling,
            accum_type=float16,
            first_order_momentum_accum_type=first_order_type,
            second_order_momentum_accum_type=float32,
        )
    elif config.optimizer == "Adam":
        optimizer = Adam(
            params,
            lr=config.learning_rate,
            weight_decay=0,
            eps=1e-6,
            loss_scaling=config.loss_scaling,
            accum_type=float16,
            first_order_momentum_accum_type=first_order_type,
            second_order_momentum_accum_type=float32,
        )
    else:
        raise ValueError("Unknown Optimizer:", config.optimizer)

    optimizer.variable_attrs.markAsConstant("weight_decay")

    # Make optimizers distributed
    if config.use_popdist:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    return optimizer
