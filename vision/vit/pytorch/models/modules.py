# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright (c) 2020 jeonsworld
# Copyright (c) Facebook, Inc. and its affiliates.
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


import copy
import math

import torch
import torch.nn as nn
from torch.nn import (Conv2d, CrossEntropyLoss, Dropout, LayerNorm, Linear,
                      Softmax)
from torch.nn.modules.utils import _pair
from typing import Tuple

import poptorch
from dataset import mixup_criterion


ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "relu": torch.nn.functional.relu,
    "tanh": torch.tanh
}


LOSS = {
    "CELoss": CrossEntropyLoss(),
}


class Representation(nn.Module):
    def __init__(self, hidden_size):
        super(Representation, self).__init__()
        self.representation_layer = nn.Linear(hidden_size, hidden_size)
        self.activation = ACT2FN["tanh"]

    def forward(self, x):
        x = self.representation_layer(x)
        x = self.activation(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    keep_mat = torch.ones(shape, dtype=x.dtype) * keep_prob
    random_tensor = torch.bernoulli(keep_mat)
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class VitEmbeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(VitEmbeddings, self).__init__()
        img_size = _pair(img_size)

        if config.patches_size is list:
            grid_size = config.patches
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
        else:
            patch_size = _pair(config.patches_size)
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.attention_probs_dropout_prob)
        self.proj_dropout = Dropout(config.hidden_dropout_prob)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.mlp_dim)
        self.fc2 = Linear(config.mlp_dim, config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.hidden_dropout_prob)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, drop_path_rate, recompute_checkpoint=False):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.recompute = recompute_checkpoint

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = self.drop_path(x)
        x = x + h

        if self.recompute:
            x = poptorch.recomputationCheckpoint(x)
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.drop_path(x)
        x = x + h
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0,
                                                config.drop_path_rate,
                                                config.num_hidden_layers)]
        for i in range(config.num_hidden_layers):
            recompute_checkpoint = i in config.recompute_mid_layers
            layer = Block(config, dpr[i], recompute_checkpoint)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_labels=21843, representation_size=None):
        super(VisionTransformer, self).__init__()
        self.config = config
        self.num_labels = num_labels
        self.representation_size = representation_size
        self.mixup = config.mixup

        self.embeddings = VitEmbeddings(config, img_size=img_size)
        self.encoder = Encoder(config)
        # additional layer in pretraining
        if representation_size is not None:
            self.representation = Representation(representation_size)
        self.head = nn.Linear(config.hidden_size, num_labels)
        self.loss = LOSS[config.loss]

    def forward(self, x, labels=None, labels_b=None, lam=None):
        if self.config.byteio:
            if self.config.precision[:3] == "16.":
                x = x.half()/255.0
            else:
                x = x.float()/255.0
        x = self.embeddings(x)
        x = self.encoder(x)
        pre_logits = x[:, 0]
        if self.representation_size is not None:
            pre_logits = self.representation(pre_logits)
        logits = self.head(pre_logits)

        if labels is None:
            return logits

        if self.mixup:
            loss = mixup_criterion(self.loss, logits, labels.view(-1),
                                   labels_b.view(-1), lam[0].item())
        else:
            loss = self.loss(logits, labels.view(-1))

        return loss, logits
