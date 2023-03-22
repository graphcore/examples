# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 OpenAI

# This file has been modified by Graphcore

from collections import OrderedDict

import numpy as np
import poptorch
import torch
from torch import nn


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def recomputation_checkpoint(module):
    """
    Annotates the output of a module to be checkpointed instead of recomputed
    """

    def recompute_outputs(module, inputs, outputs):
        if type(outputs) is tuple:
            return tuple(poptorch.recomputationCheckpoint(y) for y in outputs)
        else:
            return poptorch.recomputationCheckpoint(outputs)

    module.register_forward_hook(recompute_outputs)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, attn_mask=None)

        self.ln_post = nn.LayerNorm(width)

        # Scale the weight of self.proj
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat(
            [self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype), x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_length = config.context_length
        self.batch_size = config.batch_size
        self.memory_size = config.memory_size
        self.embed_dim = config.embed_dim

        vision_heads = config.vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=config.image_resolution,
            patch_size=config.vision_patch_size,
            width=config.vision_width,
            layers=config.vision_layers,
            heads=vision_heads,
            output_dim=config.embed_dim,
        )

        self.transformer = Transformer(
            width=config.transformer_width,
            layers=config.transformer_layers,
            heads=config.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = config.vocab_size
        self.token_embedding = torch.nn.Embedding(config.vocab_size, config.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, config.transformer_width))
        self.ln_final = nn.LayerNorm(config.transformer_width)
        self.text_projection = nn.Parameter(torch.empty(config.transformer_width, config.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

        # Allocate the register buffers to store the features of the passed steps
        self.register_buffer(
            "image_fea_queue",
            torch.normal(
                mean=0.0,
                std=(self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5),
                size=(self.memory_size * self.batch_size, self.embed_dim),
            ),
        )
        self.register_buffer(
            "text_fea_queue",
            torch.normal(
                mean=0.0,
                std=(self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5),
                size=(self.memory_size * self.batch_size, self.embed_dim),
            ),
        )

        # Loss
        self.loss = nn.CrossEntropyLoss()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    @torch.no_grad()
    def dequeue_enqueue(self, image_fea, text_fea):
        # Update the image features encoded in the current step to the register buffer
        last_image = self.image_fea_queue[: (self.memory_size - 1) * self.batch_size, :]
        update_image = torch.cat([image_fea, last_image], dim=0)
        self.image_fea_queue.copy_(update_image)

        # Update the text features encoded in the current step to the register buffer
        last_text = self.text_fea_queue[: (self.memory_size - 1) * self.batch_size, :]
        update_text = torch.cat([text_fea, last_text], dim=0)
        self.text_fea_queue.copy_(update_text)

    def build_attention_mask(self):
        # Lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -64512 which is an invalid value
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(-64512)
        mask.triu_(1)  # Zero out the lower diagonal

        return mask

    def encode_image(self, image):

        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        # Because the batch_first = False
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0], device=x.device).long(), text.argmax(dim=-1)] @ self.text_projection
        return x

    def parallelize(self, log):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Replaces self-attention layers with fused-qkv self-attention layers
        - (If enabled) Replaces the word embedding with a SerializedEmbedding
        - Adds recomputation checkpoints

        Recommended usage:
        ```
        model = CLIP(config).parallelize().half()
        ```
        """
        log.logger.info("---------- Device Allocation -----------")
        log.logger.info("image_encoder 0 --> IPU 0")
        for index in range(1):
            layer = self.visual.transformer.resblocks[index]
            recomputation_checkpoint(layer)
            self.visual.transformer.resblocks[index] = poptorch.BeginBlock(
                layer, f"image_encoder_layer{index}", ipu_id=0
            )

        log.logger.info("image_encoder 1 ~ 3 --> IPU 1")
        for index in range(1, 4):
            layer = self.visual.transformer.resblocks[index]
            recomputation_checkpoint(layer)
            self.visual.transformer.resblocks[index] = poptorch.BeginBlock(
                layer, f"image_encoder_layer{index}", ipu_id=1
            )

        log.logger.info("image_encoder 4 ~ 6 --> IPU 2")
        for index in range(4, 7):
            layer = self.visual.transformer.resblocks[index]
            recomputation_checkpoint(layer)
            self.visual.transformer.resblocks[index] = poptorch.BeginBlock(
                layer, f"image_encoder_layer{index}", ipu_id=2
            )

        log.logger.info("image_encoder 7 ~ 9 --> IPU 3")
        for index in range(7, 10):
            layer = self.visual.transformer.resblocks[index]
            recomputation_checkpoint(layer)
            self.visual.transformer.resblocks[index] = poptorch.BeginBlock(
                layer, f"image_encoder_layer{index}", ipu_id=3
            )

        log.logger.info("image_encoder 10 ~ 11 --> IPU 4")
        for index in range(10, 12):
            layer = self.visual.transformer.resblocks[index]
            recomputation_checkpoint(layer)
            self.visual.transformer.resblocks[index] = poptorch.BeginBlock(
                layer, f"image_encoder_layer{index}", ipu_id=4
            )

        log.logger.info("token_embedding --> IPU 5")
        self.token_embedding = poptorch.BeginBlock(self.token_embedding, "embedding", ipu_id=5)

        log.logger.info("text_enocder 0 --> IPU 5")
        layer = self.transformer.resblocks[0]
        recomputation_checkpoint(layer)
        self.transformer.resblocks[0] = poptorch.BeginBlock(layer, "text_encoder_layer0", ipu_id=5)

        log.logger.info("text_enocder 1 ~ 5 --> IPU 6")
        for index in range(1, 6):
            layer = self.transformer.resblocks[index]
            recomputation_checkpoint(layer)
            self.transformer.resblocks[index] = poptorch.BeginBlock(layer, f"text_encoder_layer{index}", ipu_id=6)

        log.logger.info("text_enocder 6 ~ 11 --> IPU 7")
        for index in range(6, 12):
            layer = self.transformer.resblocks[index]
            recomputation_checkpoint(layer)
            self.transformer.resblocks[index] = poptorch.BeginBlock(layer, f"text_encoder_layer{index}", ipu_id=7)

        log.logger.info("loss --> IPU 7")
        self.loss = poptorch.BeginBlock(self.loss, f"loss", ipu_id=7)
        log.logger.info("---------------------------------------")

        return self

    def forward(self, images=None, texts=None):
        if self.training:
            image_features = self.encode_image(images)
            text_features = self.encode_text(texts)

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # Concat the features encoded in the current step and from the register buffer
            n_text_features = torch.cat([text_features, self.text_fea_queue.clone().detach()], dim=0)
            n_image_features = torch.cat([image_features, self.image_fea_queue.clone().detach()], dim=0)

            logits_per_image = self.logit_scale.exp() * torch.mm(n_image_features, n_text_features.t())

            logits_per_text = logits_per_image.t()

            labels = torch.arange(logits_per_image.size()[0], device=logits_per_text.device).long()

            i_loss = self.loss(logits_per_image, labels)
            t_loss = self.loss(logits_per_text, labels)

            loss = (i_loss + t_loss) / 2.0

            self.dequeue_enqueue(image_features, text_features)

            return poptorch.identity_loss(loss, reduction="mean")

        else:
            # Only encode all the texts for zeroshot test
            if images.mean() == 0:
                class_embeddings = self.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()

                return class_embedding

            # Only encode the images for zeroshot test
            else:
                image_features = self.encode_image(images)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)

                logits_per_image = 100.0 * self.logit_scale.exp() * image_features @ texts

                return logits_per_image
