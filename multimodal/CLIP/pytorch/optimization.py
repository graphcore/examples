# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from poptorch.optim import LAMB, SGD, Adam, AdamW
from torch import float16, float32
from transformers import (get_constant_schedule,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)


def get_lr_scheduler(optimizer,
                     scheduler_type,
                     warmup_steps=None,
                     num_steps=None):
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, num_steps)
    elif scheduler_type == "constant":
        scheduler = get_constant_schedule(optimizer)
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps)
    else:
        raise ValueError("Unknown scheduler_type:", scheduler_type)

    # Prevent warning about not calling optimizer.step()
    optimizer._step_count = 1
    return scheduler


def get_optimizer(config, model):
    def exclude(n, p):
        return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n


    def include(n, p):
        return not exclude(n, p)


    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    params = [
        {"params": gain_or_bias_params, "weight_decay": 0.},
        {"params": rest_params, "weight_decay": config.weight_decay}
    ]

    if config.optimizer == "AdamW":
        optimizer = AdamW(params,
                          lr=config.learning_rate,
                          weight_decay=config.weight_decay,
                          betas=(config.beta1, config.beta2),
                          eps=config.eps,
                          loss_scaling=config.loss_scaling,
                          accum_type=float16,
                          first_order_momentum_accum_type=float32)

    elif config.optimizer == "Adam":
        optimizer = Adam(params,
                         lr=config.learning_rate,
                         weight_decay=config.weight_decay,
                         betas=(config.beta1, config.beta2),
                         eps=config.eps,
                         accum_type=float16)
    elif config.optimizer == "LAMBNoBias":
        optimizer = LAMB(params,
                         lr=config.learning_rate,
                         weight_decay=0,
                         eps=1e-6,
                         loss_scaling=config.loss_scaling,
                         max_weight_norm=None,
                         accum_type=float16,
                         bias_correction=False)
    elif config.optimizer == "LAMB":
        optimizer = LAMB(params,
                         lr=config.learning_rate,
                         weight_decay=0,
                         eps=1e-6,
                         loss_scaling=config.loss_scaling,
                         max_weight_norm=None,
                         accum_type=float16,
                         bias_correction=True)
    elif config.optimizer == "SGD":
        optimizer = SGD(params,
                        lr=config.learning_rate,
                        momentum=config.momentum,
                        weight_decay=config.weight_decay,
                        loss_scaling=config.loss_scaling,
                        use_combined_accum=True)
    else:
        raise ValueError("Unknown Optimizer:", config.optimizer)
    return optimizer
