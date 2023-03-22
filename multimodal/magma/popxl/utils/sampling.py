# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
# Copyright (c) 2022 Aleph Alpha GmbH

from magma.sampling import top_k_filter, top_p_filter
import torch
import torch.nn.functional as F


def generate(logits: torch.Tensor, top_k: float = 0.0, top_p: float = 0.9, temperature: float = 0.7):
    # taken from https://github.com/Aleph-Alpha/magma/blob/master/magma/sampling.py
    if temperature == 0.0:
        next_token = torch.argmax(logits, dim=-1, keepdims=True)
    else:
        if top_k > 0:
            logits = top_k_filter(logits, k=top_k)
        if top_p > 0:
            logits = top_p_filter(logits, threshold=top_p)

        probs = F.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    return next_token
