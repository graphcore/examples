# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention


def fix_global_eps_layer_normal_forword(self, input: torch.Tensor) -> torch.Tensor:
    global_eps = 1e-6
    return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, global_eps)


def multi_head_self_attention_forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
    """
    Parameters:
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)
    Returns:
        weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
        seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
    """
    bs, q_length, dim = query.size()
    k_length = key.size(1)

    dim_per_head = self.dim // self.n_heads

    mask_reshp = (bs, 1, 1, k_length)

    def shape(x):
        """separate heads"""
        return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

    def unshape(x):
        """group heads"""
        return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

    q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

    q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)

    # torch.finfo(torch.float16).min = -65504  use -64512 just follow experience
    scores.masked_fill_(mask, -64512)  # (bs, n_heads, q_length, k_length)

    weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
    weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

    # Mask heads if we want to
    if head_mask is not None:
        weights = weights * head_mask

    context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    context = unshape(context)  # (bs, q_length, dim)
    context = self.out_lin(context)  # (bs, q_length, dim)

    if output_attentions:
        return (context, weights)
    else:
        return (context,)


def do_patch():
    # Fix global eps of layer normal
    torch.nn.LayerNorm.forward = fix_global_eps_layer_normal_forword
    # Fix masked_fill_
    MultiHeadSelfAttention.forward = multi_head_self_attention_forward
