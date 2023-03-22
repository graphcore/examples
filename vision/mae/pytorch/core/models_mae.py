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
#
# This file has been modified by Graphcore Ltd.
from functools import partial
import torch
import torch.nn as nn
import poptorch
from core.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed
from core.gelu import ERF_GELU
from util.log import logger


def remap_tensor(
    x,
    fwd_grain_size=8,
    bwd_grain_size=0,
    fwd_clone_layout=False,
    bwd_clone_layout=False,
    fwd_after_matmul=False,
    bwd_after_matmul=False,
    debug_str="",
):
    if 0 == bwd_grain_size:
        bwd_grain_size = fwd_grain_size
    return poptorch.custom_op(
        [x],
        "RemapCE",
        "ai.graphcore",
        1,
        example_outputs=[x],
        attributes={
            "fwd_grain_size": fwd_grain_size,
            "bwd_grain_size": bwd_grain_size,
            "fwd_clone_layout": 1 if fwd_clone_layout is True else 0,
            "bwd_clone_layout": 1 if bwd_clone_layout is True else 0,
            "fwd_after_matmul": 1 if fwd_after_matmul is True else 0,
            "bwd_after_matmul": 1 if bwd_after_matmul is True else 0,
            "debug_str": debug_str,
        },
    )[0]


class MaeLoss(nn.Module):
    def __init__(self, norm_pix_loss, p):
        super().__init__()
        self.p = p
        self.norm_pix_loss = norm_pix_loss
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.var = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # mean and variance should be on the same device
        self.mean = self.mean.to(imgs.device)
        self.var = self.var.to(imgs.device)

        imgs = imgs.detach() / 255.0
        imgs = (imgs - self.mean) / self.var
        pred = pred.float()
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)

            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.p
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        gelu_type="erf",
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        pipeline=None,
        device="ipu",
        mask_ratio=0.75,
        half=False,
    ):
        super().__init__()

        # MAE encoder specifics
        self.use_half = half
        gelu = nn.GELU
        if gelu_type == "erf":
            gelu = ERF_GELU
        self.mask_ratio = mask_ratio
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, act_layer=gelu)
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    act_layer=gelu,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)  # decoder to patch

        self.initialize_weights()
        self.loss = MaeLoss(norm_pix_loss, self.patch_embed.patch_size)

        # ipu set
        self.device = device
        self.pipeline = pipeline
        if self.pipeline is not None:
            self._set_pipeline()

    def _get_layer_ipu(self, pipeline):
        layer_ipu = []
        for ipu, n_layers in enumerate(pipeline):
            layer_ipu += [ipu] * n_layers
        return layer_ipu

    def _set_mae(self, embed, blocks, norm, layer_ipu, first_id, block_count, stage):
        embed = poptorch.BeginBlock(embed, f"ipu{stage}", ipu_id=first_id)
        logger.info(f"embed on ipu {first_id}, stage:{stage}")
        for i, block in enumerate(blocks):
            index = block_count + i
            logger.info(index)
            ipu = layer_ipu[index]
            logger.info(f"begin block at ({i}) on ipu:{ipu}, stage:{stage}")
            blocks[i] = poptorch.BeginBlock(block, f"ipu{stage}", ipu_id=ipu)
            stage += 1
        norm = poptorch.BeginBlock(norm, f"ipu{stage-1}", ipu_id=ipu)
        logger.info(f"norm on ipu {ipu}, stage:{stage-1}")
        stage += 1
        return ipu, index + 1, stage

    def _set_pipeline(self):
        stage = 0
        num = len(self.pipeline) / 2
        pipeline = self.pipeline[: int(num)]
        layer_ipu = self._get_layer_ipu(pipeline)
        last_id, block_count, stage = self._set_mae(self.patch_embed, self.blocks, self.norm, layer_ipu, 0, 0, stage)
        pipeline = self.pipeline[int(num) :]
        layer_ipu = self._get_layer_ipu(pipeline)
        last_id, _, stage = self._set_mae(
            self.decoder_embed, self.decoder_blocks, self.decoder_norm, layer_ipu, 0, 0, stage
        )
        self.decoder_pred = poptorch.BeginBlock(self.decoder_pred, f"ipu{stage}", ipu_id=last_id)
        logger.info(f"decoder_pred on ipu {last_id}, stage: {stage}")
        self.loss = poptorch.BeginBlock(self.loss, f"ipu{stage}", ipu_id=last_id)
        logger.info(f"loss on ipu {last_id}, stage: {stage}")

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as
        # cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def random_masking(self, x, mask_ratio, ids_shuffle, keep_mat):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # keep the first subset
        # workaround for gather
        if self.device == "ipu":
            x_masked = x.permute(0, 2, 1).matmul(keep_mat).permute(0, 2, 1)
        else:
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked

    def forward_encoder(self, x, mask_ratio, ids_shuffle, keep_mat):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x = self.random_masking(x, mask_ratio, ids_shuffle, keep_mat)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        if self.use_half:
            x = torch.cat((cls_tokens.half(), x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            if self.device == "ipu":
                x = remap_tensor(x)
                x = blk(x)
                x = remap_tensor(x)
            else:
                x = blk(x)
        x = self.norm(x)
        return x

    def forward_decoder(self, x, ids_restore, restore_mat):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        if self.use_half:
            x_ = torch.cat([x[:, 1:, :], mask_tokens.half()], dim=1)  # no cls token
        else:
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        if self.device == "ipu":
            x_ = x_.permute(0, 2, 1).matmul(restore_mat).permute(0, 2, 1)
        else:
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed.detach()

        # apply Transformer blocks

        for blk in self.decoder_blocks:
            if self.device == "ipu":
                x = remap_tensor(x)
                x = blk(x)
                x = remap_tensor(x)
            else:
                x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward(self, imgs, target, ids_restore, keep_mat, restore_mat, mask, ids_shuffle=None):
        imgs = imgs / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        var = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        imgs = (imgs - mean) / var
        if self.use_half:
            imgs = imgs.half()

        latent = self.forward_encoder(imgs, self.mask_ratio, ids_shuffle, keep_mat)
        pred = self.forward_decoder(latent, ids_restore, restore_mat)  # [N, L, p*p*3]
        loss = self.loss(target, pred, mask)
        return pred, poptorch.identity_loss(loss, reduction="mean")


def mae_vit_align_patch16_dec512d4b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=3,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=2,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# set recommended archs
mae_vit_align_patch16 = mae_vit_align_patch16_dec512d4b
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
