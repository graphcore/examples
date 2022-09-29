# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright 2022 Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Graphcore Ltd.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import trunc_normal_, Precision
import poptorch
from core.weight_norm import weight_norm


class DINOLoss(nn.Module):
    def __init__(self, ncrops, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops

    def forward(
            self,
            student_output,
            teacher_output,
            center,
            teacher_temp_factor):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        teacher_out = F.softmax(
            (teacher_output - center) / teacher_temp_factor, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the
                    # same view
                    continue
                loss = torch.sum(-q *
                                 F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        return batch_center, total_loss


class DINOHead(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            use_bn=False,
            norm_last_layer=True,
            nlayers=3,
            hidden_dim=2048,
            bottleneck_dim=256,
            act_layer=nn.GELU,
            precision=Precision.FP32):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.precision = precision
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_layer())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(act_layer())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer1 = weight_norm(
            nn.Linear(
                bottleneck_dim,
                out_dim // 2,
                bias=False))
        self.last_layer1.weight_g.data.fill_(1)
        self.last_layer2 = weight_norm(
            nn.Linear(
                bottleneck_dim,
                out_dim // 2,
                bias=False))
        self.last_layer2.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        if self.precision is not Precision.FP32:
            x = nn.functional.normalize(x.float(), dim=-1, p=2).half()
        else:
            x = nn.functional.normalize(x, dim=-1, p=2)
        x1 = self.last_layer1(x)
        x2 = self.last_layer2(x)
        x = torch.cat((x1, x2), axis=-1)
        return x


def _get_layer_ipu(layers_per_ipu):
    # List of the IPU Id for each encoder layer
    layer_ipu = []
    for ipu, n_layers in enumerate(layers_per_ipu):
        layer_ipu += [ipu] * n_layers
    return layer_ipu


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self,
                 student,
                 teacher,
                 student_head,
                 teacher_head,
                 loss,
                 momentum,
                 device='ipu',
                 pipeline=None,
                 precision=Precision.FP32,
                 alignment=False):
        super(MultiCropWrapper, self).__init__()
        self.student = student
        self.student_head = student_head
        self.teacher = teacher
        self.teacher_head = teacher_head
        self.loss = loss

        self.momentum = momentum

        self.device = device
        self.pipeline = pipeline
        self.alignment = alignment
        if pipeline is not None:
            self.set_pipeline()
        self.precision = precision

        teacher.load_state_dict(student.state_dict())
        teacher_head.load_state_dict(student_head.state_dict())

        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

    def forward(
            self,
            global_imgs,
            crops,
            ema_factor,
            center,
            teacher_temp_factor):
        bs, global_count = global_imgs.shape[:2]
        crop_count = crops.shape[1]
        global_imgs = torch.chunk(global_imgs, global_count, dim=1)
        crops = torch.chunk(crops, crop_count, dim=1)
        global_imgs = torch.cat([img.squeeze(1) for img in global_imgs])
        crops = torch.cat([img.squeeze(1) for img in crops])
        global_imgs = global_imgs / 255.
        crops = crops / 255.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        var = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        global_imgs = (global_imgs - mean) / var
        crops = (crops - mean) / var
        if self.precision is not Precision.FP32:
            global_imgs = global_imgs.half()
            crops = crops.half()

        student_output_global = self.student(global_imgs)
        student_output_crop = self.student(crops)

        student_output = torch.cat(
            (student_output_global, student_output_crop))
        student_output = self.student_head(student_output)

        with torch.no_grad():
            self._momentum_update_teacher(ema_factor)
            teacher_output = self.teacher(global_imgs)
            teacher_output = self.teacher_head(teacher_output)
            teacher_output = teacher_output.detach()

        batch_center, loss = self.loss(
            student_output.float(), teacher_output.float(), center, teacher_temp_factor)
        if self.alignment:
            return torch.cat([student_output, teacher_output],
                             dim=0), poptorch.identity_loss(loss, reduction='mean')
        else:
            return batch_center, poptorch.identity_loss(loss, reduction='mean')

    @torch.no_grad()
    def gpu_update_teacher(self):
        for param_q, param_k in zip(
                self.student.parameters(), self.teacher.parameters()):
            param_k.data.mul_(
                self.momentum).add_(
                (1 - self.momentum) * param_q.detach().data)
        for param_q, param_k in zip(
                self.student_head.parameters(), self.teacher_head.parameters()):
            param_k.data.mul_(
                self.momentum).add_(
                (1 - self.momentum) * param_q.detach().data)

    @torch.no_grad()
    def _momentum_update_teacher(self, ema_factor):
        """
        Momentum update of the key encoder
        """
        if self.device == 'ipu':
            self._momentum_update_pipeline(ema_factor)

    def set_vit(self, vit, layer_ipu, index_list):
        vit.patch_embed = poptorch.BeginBlock(
            vit.patch_embed, 'ipu0', ipu_id=0)
        for i, block in enumerate(vit.blocks):
            ipu = layer_ipu[i]
            if i in index_list:
                vit.blocks[i] = poptorch.BeginBlock(
                    block, f'ipu{ipu}', ipu_id=ipu)

    def set_pipeline(self):
        layer_ipu = _get_layer_ipu(self.pipeline)
        last_id = len(self.pipeline) - 1
        index_list = []
        for i, v in enumerate(self.pipeline):
            index = sum(self.pipeline[:i])
            index_list.append(index)
        self.set_vit(self.student, layer_ipu, index_list)
        self.set_vit(self.teacher, layer_ipu, index_list)

        self.student_head.mlp = poptorch.BeginBlock(
            self.student_head.mlp, f'ipu{last_id-1}', ipu_id=last_id - 1)
        self.teacher_head.mlp = poptorch.BeginBlock(
            self.teacher_head.mlp, f'ipu{last_id-1}', ipu_id=last_id - 1)

        self.student_head.last_layer1 = poptorch.BeginBlock(
            self.student_head.last_layer1, f'ipu{last_id}', ipu_id=last_id)
        self.teacher_head.last_layer1 = poptorch.BeginBlock(
            self.teacher_head.last_layer1, f'ipu{last_id}', ipu_id=last_id)
        self.student_head.last_layer2 = poptorch.BeginBlock(
            self.student_head.last_layer2, f'ipu{last_id}', ipu_id=last_id)
        self.teacher_head.last_layer2 = poptorch.BeginBlock(
            self.teacher_head.last_layer2, f'ipu{last_id}', ipu_id=last_id)
        self.loss = poptorch.BeginBlock(
            self.loss, f'ipu{last_id}', ipu_id=last_id)

    @torch.no_grad()
    def _momentum_update_pipeline(self, ema_factor):
        layer_ipu = _get_layer_ipu(self.pipeline)
        ipu_count = len(self.pipeline)
        last_id = len(self.pipeline) - 1

        self.ema_update(
            self.student.cls_token,
            self.teacher.cls_token,
            ema_factor,
            0)
        self.ema_update(
            self.student.pos_embed,
            self.teacher.pos_embed,
            ema_factor,
            0)
        self.single_ema_update(
            self.student.patch_embed,
            self.teacher.patch_embed,
            ema_factor,
            0)

        for i, layer in enumerate(self.student.blocks):
            ipu = layer_ipu[i]
            self.single_ema_update(
                self.student.blocks[i],
                self.teacher.blocks[i],
                ema_factor,
                ipu)
        self.single_ema_update(
            self.student.norm,
            self.teacher.norm,
            ema_factor,
            ipu)
        self.single_ema_update(
            self.student_head.mlp,
            self.teacher_head.mlp,
            ema_factor,
            last_id - 1)
        self.single_ema_update(
            self.student_head.last_layer1,
            self.teacher_head.last_layer1,
            ema_factor,
            last_id)
        self.single_ema_update(
            self.student_head.last_layer2,
            self.teacher_head.last_layer2,
            ema_factor,
            last_id)

    @torch.no_grad()
    def single_ema_update(self, modeule_q, module_k, ema_factor, ipu_id):
        for param_q, param_k in zip(
                modeule_q.parameters(), module_k.parameters()):
            self.ema_update(param_q, param_k, ema_factor, ipu_id)

    @torch.no_grad()
    def ema_update(self, param_q, param_k, ema_factor, ipu_id):
        with poptorch.Block(f'ipu{ipu_id}', ipu_id):
            poptorch.custom_op([param_q, param_k, ema_factor],
                               'ExpMovAvg',
                               'com.acme',
                               1,
                               example_outputs=[param_q],
                               attributes={})
