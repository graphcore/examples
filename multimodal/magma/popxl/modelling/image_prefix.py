# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from popxl_addons import Module, NamedTensors
from popxl_addons.layers import Linear, LayerNorm
import popxl
from popxl import Tensor
from popxl import ops
from popxl.utils import to_numpy

from configs import MagmaConfig
from modelling.clip_resnet.modified_resnet import ModifiedResNet
import numpy as np


class ImagePrefix(Module):
    def __init__(self, config: MagmaConfig):
        """
        Takes in a batch of images and returns a batch of embeddings of the
        same dimensions as the LM's word embeddings.
        """
        super().__init__()
        self.config = config
        proj_out_dim = self.config.transformer.hidden_size
        # project to the language model hidden_dim
        self.proj = Linear(proj_out_dim)
        self.enc = ModifiedResNet(config.visual, pool=False)
        self.ln = LayerNorm()

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        # pass through image encoder
        #: (b, channels, h, w)
        embed = self.enc(x)
        #: b h w d
        assert len(embed.shape) == 4
        if embed.shape[1] == 1:
            embed = ops.squeeze(embed, [1, 2])
        else:
            #: b (h w) d
            embed = embed.reshape((*embed.shape[:2], embed.shape[2] * embed.shape[3])).transpose((0, 2, 1))
        #: b (h w) d -> b ( h w ) proj_out_dim = b ( h w ) lm_hidden_size
        embed = self.proj(embed)
        bs, hw, hidden = embed.shape
        embed = embed.reshape((bs * hw, hidden))
        embed = self.ln(embed)
        embed = embed.reshape((bs, hw, hidden))
        return embed

    @staticmethod
    def magma_mapping(magma_model, config, variables: NamedTensors):
        state_dict = ModifiedResNet.clip_mapping(magma_model.enc, config.visual, variables.enc, False)
        state_dict.update(
            {
                variables.ln.weight: to_numpy(magma_model.ln.weight),
                variables.ln.bias: to_numpy(magma_model.ln.bias),
                variables.proj.weight: to_numpy(magma_model.proj.weight.T),
                variables.proj.bias: to_numpy(magma_model.proj.bias),
            }
        )
        return state_dict
