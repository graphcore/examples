# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import poptorch

import timm.models.vision_transformer
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from core.gelu import ERF_GELU
from util.log import logger


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(
            self,
            global_pool=False,
            pipeline=None,
            criterion=None,
            act_layer=nn.GELU,
            **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.pipeline = pipeline
            self.criterion = criterion
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        self.pipeline = pipeline
        if self.pipeline is not None:
            self._set_pipeline()

    def _get_layer_ipu(self, pipeline):
        layer_ipu = []
        for ipu, n_layers in enumerate(pipeline):
            layer_ipu += [ipu] * n_layers
        return layer_ipu

    def _set_vit(self, embed, blocks, norm, layer_ipu, first_id, block_count):
        embed = poptorch.BeginBlock(embed, f'ipu{first_id}', ipu_id=first_id)
        logger.info(f'embed on ipu {first_id}')
        for i, block in enumerate(blocks):
            index = block_count + i
            ipu = layer_ipu[index]
            logger.info(f'begin block at ({i}) on ipu:{ipu}')
            blocks[i] = poptorch.BeginBlock(block, f'ipu{ipu}', ipu_id=ipu)
        norm = poptorch.BeginBlock(norm, f'ipu{ipu}', ipu_id=ipu)
        logger.info(f'norm on ipu {ipu}')
        return ipu, index + 1

    def _set_pipeline(self):
        layer_ipu = self._get_layer_ipu(self.pipeline)
        last_id, block_count = self._set_vit(
            self.patch_embed, self.blocks, self.fc_norm, layer_ipu, 0, 0)
        self.head = poptorch.BeginBlock(self.head, f'ipu{last_id}', ipu_id=last_id)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

    def forward(self, x, targets=None):
        x = self.forward_features(x)
        x = self.head(x)
        if targets is None:
            return x
        else:
            loss = self.criterion(x, targets)
            return x, poptorch.identity_loss(loss, reduction='mean')


def vit_base_patch16(criterion, pipeline=None, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm,
            eps=1e-6),
        criterion=criterion,
        pipeline=pipeline,
        **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm,
            eps=1e-6),
        **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm,
            eps=1e-6),
        **kwargs)
    return model
