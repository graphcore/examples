# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 Max Bain
# This file has been modified by Graphcore

import os
from collections import OrderedDict

import numpy as np

import poptorch
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .multi_head_self_attention import do_patch
from .video_transformer import SpaceTimeTransformer

do_patch()


def sim_matrix_original(a, b, eps=1e-6):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def sim_matrix(a, b):
    # Normalized features
    a = (a / a.norm(dim=-1, keepdim=True)).type(torch.float32)
    b = (b / b.norm(dim=-1, keepdim=True)).type(torch.float32)
    sim_mt = a @ b.t()
    return sim_mt


def sim_matrix_with_epsilon(a, b, eps=1e-4):
    # Normalized features
    a_n, b_n = a.norm(dim=-1, keepdim=True), b.norm(dim=-1, keepdim=True)
    a = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = a @ b.t()
    return sim_mt


def state_dict_data_parallel_fix(load_state_dict, curr_state_dict):
    # Load state dict
    load_keys = list(load_state_dict.keys())
    curr_keys = list(curr_state_dict.keys())

    redo_dp = False
    undo_dp = False
    if not curr_keys[-1].startswith("model.") and load_keys[-1].startswith("model."):
        undo_dp = True
    elif curr_keys[-1].startswith("model.") and not load_keys[-1].startswith("model."):
        redo_dp = True

    if undo_dp:
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            if k.startswith("model."):
                k = k[6:]
                new_state_dict[k] = v
        # Load params
    elif redo_dp:
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = "model." + k
            new_state_dict[name] = v
    else:
        new_state_dict = load_state_dict
    return new_state_dict


def recomputation_checkpoint(module):
    def recompute_outputs(module, inputs, outputs):
        if type(outputs) is tuple:
            return tuple(poptorch.recomputationCheckpoint(y) for y in outputs)
        else:
            return poptorch.recomputationCheckpoint(outputs)

    module.register_forward_hook(recompute_outputs)


class FrozenInTime(nn.Module):
    def __init__(self, video_params, text_params, projection_dim=256, load_checkpoint=None, load_temporal_fix="zeros"):
        super().__init__()

        self.video_params = video_params
        self.text_params = text_params
        self.load_temporal_fix = load_temporal_fix

        text_model_name = text_params.get("model", "distilbert-base-uncased")
        if "distilbert-base-uncased" in text_model_name:
            self.text_model = AutoModel.from_pretrained(
                text_model_name,
                n_layers=text_params.get("num_layers", 6),
                dropout=text_params.get("dropout", 0.1),
                attention_dropout=text_params.get("attention_dropout", 0.1),
            )
        else:
            raise NotImplementedError(f"{text_model_name} not implemented so far.")

        pretrained = video_params["pretrained"]

        num_frames = video_params.get("num_frames", 4)
        time_init = video_params.get("time_init", "zeros")
        arch_config = video_params.get("arch_config", "base_patch16_224")
        num_layers = video_params.get("num_layers", 12)
        if arch_config == "base_patch16_224":
            patch_size = 16
            vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
        elif arch_config == "base_patch32_224":
            patch_size = 32
            vit_model = timm.models.vision_transformer.vit_base_patch32_224(pretrained=pretrained)
        elif arch_config == "openai/clip-vit-base-patch32":
            raise NotImplementedError(f"{arch_config} not implemented so far.")
            patch_size = 32
            vit_model = AutoModel.from_pretrained(arch_config).vision_model
        else:
            raise NotImplementedError(f"{arch_config} not implemented")

        model = SpaceTimeTransformer(
            patch_size=patch_size, num_frames=num_frames, time_init=time_init, depth=num_layers
        )

        model.head = nn.Identity()
        model.pre_logits = nn.Identity()
        ftr_dim = model.embed_dim
        if load_checkpoint in ["", None]:
            vit_checkpoint = vit_model.state_dict()
            model.load_state_dict(vit_checkpoint, strict=False)
        self.video_model = model

        # For backwards compatibility (old models)
        self.video_model.fc = nn.Identity()

        # Project to a common embedding
        self.txt_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.text_model.config.hidden_size, projection_dim),
        )
        self.vid_proj = nn.Sequential(nn.Linear(ftr_dim, projection_dim))

        if os.path.exists(load_checkpoint):
            checkpoint = torch.load(load_checkpoint)
            state_dict = checkpoint["state_dict"]
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            self.load_state_dict(new_state_dict, strict=True)

    def forward(self, input_ids, attention_mask, video):
        text_embeddings = self.compute_text(input_ids, attention_mask)
        video_embeddings = self.compute_video(video)

        return text_embeddings, video_embeddings

    def compute_text(self, input_ids, attention_mask):
        text_embeddings = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_video(self, video_data):
        video_embeddings = self.video_model(video_data)
        video_embeddings = self.vid_proj(video_embeddings)
        return video_embeddings

    def _inflate_positional_embeds(self, new_state_dict):
        # Allow loading of TimeSforme with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if "video_model.temporal_embed" in new_state_dict and "video_model.temporal_embed" in curr_keys:
            load_temporal_embed = new_state_dict["video_model.temporal_embed"]
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params["num_frames"]
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    self.logger.info(
                        f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                        f"### loading weights, filling in the extras via {self.load_temporal_fix}"
                    )
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    self.logger.info(
                        f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                        f"### loading weights, filling in the extras via {self.load_temporal_fix}"
                    )
                    if self.load_temporal_fix == "zeros":
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ["interp", "bilinear"]:
                        # Unsqueeze so pytorch thinks its an image
                        mode = "bilinear" if self.load_temporal_fix == "bilinear" else "nearest"
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(
                            load_temporal_embed, (curr_num_frames, embed_dim), mode=mode
                        ).squeeze(0)
                    else:
                        raise NotImplementedError("Temporal model invalid. Support zeros, interp or bilinear")
                new_state_dict["video_model.temporal_embed"] = new_temporal_embed
        # Allow loading with smaller spatial patches. assumes custom border crop, to append the border patches to the input sequence
        if "video_model.pos_embed" in new_state_dict and "video_model.pos_embed" in curr_keys:
            load_pos_embed = new_state_dict["video_model.pos_embed"]
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()["video_model.pos_embed"]
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    "Loading models with different spatial resolution / patch number not yet implemented, sorry."
                )

        return new_state_dict

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum(np.prod(p.size()) for p in model_parameters)
        # Return super().__str__() + '\nTrainable parameters: {}'.format(params)
        return "\nTrainable parameters: {}".format(params)


class PipelinedWithLoss(nn.Module):
    def __init__(self, model, loss, logger):
        """
        Recommended usage: model = PipelinedWithLoss(model, loss).half()
        """
        super().__init__()
        self.loss = loss
        self.logger = logger
        self.model = model

    def parallelize(self, pipelined_layers: dict):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Adds recomputation checkpoints
        """
        self.logger.info("---------- parallelize in pod8 ---------")

        recomputation_checkpoint(self.model.text_model.embeddings)
        ipu_id = pipelined_layers.get("txt_embeddings", 0)
        self.model.text_model.embeddings = poptorch.BeginBlock(
            self.model.text_model.embeddings, "txt_embeddings", ipu_id=ipu_id
        )
        self.logger.info(f"txt_embeddings --> IPU {ipu_id}")

        layers_on_ipu = pipelined_layers.get("txt_transformer", [1, 1, 1, 2, 2, 2])
        for index, layer in enumerate(self.model.text_model.transformer.layer):
            recomputation_checkpoint(layer)
            self.model.text_model.transformer.layer[index] = poptorch.BeginBlock(
                layer, f"txt_transformer {index}", ipu_id=layers_on_ipu[index]
            )
            self.logger.info(f"txt_transformer {index} --> IPU {layers_on_ipu[index]}")

        recomputation_checkpoint(self.model.txt_proj)
        ipu_id = pipelined_layers.get("txt_proj", 2)
        self.model.txt_proj = poptorch.BeginBlock(self.model.txt_proj, "txt_proj", ipu_id=ipu_id)
        self.logger.info(f"txt_proj --> IPU {ipu_id}")

        recomputation_checkpoint(self.model.video_model.patch_embed)
        ipu_id = pipelined_layers.get("vid_patch_embed", 3)
        self.model.video_model.patch_embed = poptorch.BeginBlock(
            self.model.video_model.patch_embed, "vid_patch_embed", ipu_id=ipu_id
        )
        self.logger.info(f"vid_patch_embed --> IPU {ipu_id}")

        layers_on_ipu = pipelined_layers.get("vid_blocks", [4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7])
        for index, layer in enumerate(self.model.video_model.blocks):
            recomputation_checkpoint(layer)
            self.model.video_model.blocks[index] = poptorch.BeginBlock(
                layer, f"vid_blocks {index}", ipu_id=layers_on_ipu[index]
            )
            self.logger.info(f"vid_blocks {index} --> IPU {layers_on_ipu[index]}")

        recomputation_checkpoint(self.model.vid_proj)
        ipu_id = pipelined_layers.get("vid_proj", 7)
        self.model.vid_proj = poptorch.BeginBlock(self.model.vid_proj, "vid_proj", ipu_id=ipu_id)
        self.logger.info(f"vid_proj --> IPU {ipu_id}")

        self.logger.info("---------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, video):
        text_embeddings, video_embeddings = self.model(input_ids, attention_mask, video)

        if self.training:
            sim_mtrx = sim_matrix(text_embeddings, video_embeddings)
            return poptorch.identity_loss(self.loss(sim_mtrx), reduction="none")
        else:
            results = (text_embeddings, video_embeddings)
            return results
