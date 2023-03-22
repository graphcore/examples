# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import numpy as np
import math

from popxl_addons import Module, NamedTensors
from popxl_addons.layers import Linear
import popxl
from popxl import Tensor
from popxl import ops
from popxl.utils import to_numpy
from clip.model import AttentionPool2d
from configs import ResNetConfig

__all__ = ["AttentionPool"]


class AttentionPool(Module):
    def __init__(self, config: ResNetConfig):
        """
        Use self-attention mechanism to perform pooling and reduce the input from (N,C,H,W) to (N,C)
        From CLIP paper:
            The attention pooling is implemented as a single layer of “transformer-style” multi-head QKV attention
            where the query is conditioned on the global average-pooled
        """
        super().__init__()
        self.config = config
        self.spatial_dim = self.config.image_resolution // 32
        self.head_dim = self.config.embed_dim // self.config.heads

        self.k_proj = Linear(self.config.embed_dim)
        self.q_proj = Linear(self.config.embed_dim)
        self.v_proj = Linear(self.config.embed_dim)
        self.c_proj = Linear(self.config.embed_dim)

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        #: x (batch, embed dim, spatial dim, spatial dim)
        #: N batch
        #: C channel -> equal to embed dimension
        #: H height -> equal to spatial dim
        #: W width -> equal to spatial dim

        n, c, h, w = x.shape
        x = x.reshape((n, c, h * w)).transpose((2, 0, 1))  # NCHW -> (HW)NC
        # global average pooling
        x = ops.concat([ops.mean(x, axis=0, keepdims=True), x], axis=0)  # (HW+1)NC

        def initialiser():
            return np.random.rand(self.spatial_dim**2 + 1, 1, self.config.embed_dim) / self.config.embed_dim**0.5

        self.positional_embedding = self.add_variable_input("positional_embedding", initialiser, self.config.dtype)
        x = x + self.positional_embedding  # (HW+1)NC
        query = x[:1]  # query is global average pooling
        key = x
        value = x
        tgt_len, bsz, embed_dim = query.shape
        # ----- SELF ATTENTION -----
        # no dropout

        #: projection
        query = self.q_proj(query)  #: (HW,embed_dim)
        # then split (fused attention) for key and value since they are equal
        key = self.k_proj(key)  #: (HW + 1, embed_dim)
        value = self.v_proj(value)  #: (HW + 1), embed_dim

        #: reshape to separate each head and have batch dimension first
        def split_heads(x: popxl.Tensor, is_key=False):
            if is_key:
                return x.reshape((x.shape[0], bsz * self.config.heads, self.head_dim)).transpose((1, 2, 0))
            else:
                return x.reshape((x.shape[0], bsz * self.config.heads, self.head_dim)).transpose((1, 0, 2))

        query = split_heads(query)  #: (bsz*num_heads, 1, head_dim)
        key = split_heads(key, True)  #: (bsz*num_heads, head_dim, HW + 1)
        value = split_heads(value)  #: (bsz*num_heads, HW + 1, head_dim)

        #: attention scores
        scale = 1.0 / math.sqrt(query.shape[-1])
        scores = query @ key * scale  #: (bsz*num_heads, 1, HW + 1)
        scores = ops.softmax(scores, axis=-1)

        #: attention output
        #: scores (bsz*num_heads, 1, HW + 1)
        #: value (bsz*num_heads, HW + 1, head_dim)
        attn_output = scores @ value  #: (bsz*num_heads, 1, head_dim)
        # back to full embedding dim (merge heads)
        #: transpose -> (1, bsz*num_heads, head_dim)
        #: reshape -> (1, bsz, head_dim*num_heads)
        attn_output = attn_output.transpose((1, 0, 2)).reshape((tgt_len, bsz, embed_dim))  #: (1, bsz, embed_dim)
        attn_output = self.c_proj(attn_output)  #: (1, bsz, output_dim) = (1, bsz, embed_dim)
        attn_output = ops.squeeze(attn_output, [0])
        return attn_output

    @staticmethod
    def clip_mapping(clip_model: AttentionPool2d, variables: NamedTensors):
        pos_embedding_shape = (clip_model.positional_embedding.shape[0], 1, clip_model.positional_embedding.shape[1])
        state_dict = {
            variables.positional_embedding: to_numpy(clip_model.positional_embedding.data.reshape(pos_embedding_shape)),
            variables.k_proj.weight: to_numpy(clip_model.k_proj.weight.data.T),
            variables.k_proj.bias: to_numpy(clip_model.k_proj.bias.data),
            variables.q_proj.weight: to_numpy(clip_model.q_proj.weight.data.T),
            variables.q_proj.bias: to_numpy(clip_model.q_proj.bias.data),
            variables.v_proj.weight: to_numpy(clip_model.v_proj.weight.data.T),
            variables.v_proj.bias: to_numpy(clip_model.v_proj.bias.data),
            variables.c_proj.weight: to_numpy(clip_model.c_proj.weight.data.T),
            variables.c_proj.bias: to_numpy(clip_model.c_proj.bias.data),
        }
        return state_dict
