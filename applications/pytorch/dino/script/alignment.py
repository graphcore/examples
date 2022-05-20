# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
import os
import sys
import math
import json
import ctypes
import time
import datetime
from pathlib import Path
import numpy as np
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn
import poptorch
from poptorch.optim import SGD, AdamW
sys.path.append('..')
from core.gelu import ERF_GELU
from core import utils
from core import vision_transformer as vits
from core.dino import DINOLoss, DINOHead, MultiCropWrapper
from options import alignment_options, get_options


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument(
        '--arch',
        default='vit_base',
        type=str,
        choices=[
            'vit_tiny',
            'vit_small',
            'vit_base',
            'vit_mini'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument(
        '--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument(
        '--out_dim', default=1024, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument(
        '--norm_last_layer',
        default=True,
        type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this parameter to False with vit_small and True with vit_base.""")
    parser.add_argument(
        '--momentum_teacher', default=0.6, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument(
        '--use_bn_in_head',
        default=False,
        type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--batch_size', default=1, type=int, help='batch-size')
    parser.add_argument(
        '--drop_path_rate',
        type=float,
        default=0.,
        help="stochastic depth rate")

    parser.add_argument(
        '--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)

    parser.add_argument(
        '--pipeline',
        default=None,
        type=int,
        nargs='+',
        help='set modules on multi ipus')
    parser.add_argument('--alignment', default=True, type=utils.bool_flag)
    parser.add_argument('--extract_name', action='store_true')
    parser.add_argument('--grad_compare', action='store_true')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument(
        '--device',
        type=str,
        default='ipu',
        help='device to use')
    parser.add_argument(
        '--ema_so',
        type=str,
        default='../ema/build/exp_avg_custom_op.so',
        help='custom ema, path of so')
    parser.add_argument('--output', type=str, default='./alignment')
    parser.add_argument('--grad', type=str, default='grad_names.pt')
    parser.add_argument('--alignment_pipeline', action='store_true')
    parser.add_argument('--ga', default=12, type=int)
    return parser


def extract_name(args, model, optimizer, center):
    path = os.path.join(args.output, args.grad)
    if os.path.exists(path):
        return
    assert os.path.exists(args.ema_so), 'please compile custom op ema'
    libc = ctypes.cdll.LoadLibrary(args.ema_so)
    opts = alignment_options()
    ipu_model = poptorch.trainingModel(model, opts, optimizer=optimizer)
    img1 = torch.randint(0, 255, (1, 2, 3, 224, 224), dtype=torch.uint8)
    img2 = torch.randint(0, 255, (1, 8, 3, 96, 96), dtype=torch.uint8)
    ema_factor_base = torch.ones((1))
    ema_factor = ema_factor_base * args.momentum_teacher
    teacher_temp_factor = 0.04 * torch.ones((1))
    _, loss = ipu_model(img1, img2, ema_factor, center, teacher_temp_factor)
    tensor_names = ipu_model.getTensorNames()
    torch.save(tensor_names, path)


def load_weight(model, path):
    state_dict = torch.load(path)
    new_dict = OrderedDict()
    for n, v in model.state_dict().items():
        if n in state_dict:
            new_dict[n] = state_dict[n]
        else:
            print(f'{n} not in state dict')
            new_dict[n] = v
    model.load_state_dict(new_dict)


def shard_alignment(args, model, optimizer, center):
    img_name = os.path.join(args.output, 'image.pth')
    if os.path.exists(img_name):
        img1, img2 = torch.load(img_name)
        print('load image')
    else:
        bs = args.batch_size
        img1 = torch.randint(0, 255, (bs, 2, 3, 224, 224), dtype=torch.uint8)
        img2 = torch.randint(0, 255, (bs, 8, 3, 96, 96), dtype=torch.uint8)
        torch.save([img1, img2], img_name)

    compare_weights = False
    dir_path = os.path.join(args.output, args.device)
    os.makedirs(dir_path, exist_ok=True)
    count = 10
    grad_count = 5
    ema_factor_base = torch.ones((1))
    ema_factor = ema_factor_base * args.momentum_teacher
    global_center = center.repeat(args.batch_size, 1)
    teacher_temp_factor = 0.04 * torch.ones((args.batch_size))
    if args.device == 'ipu':
        assert os.path.exists(args.ema_so), 'please compile custom op ema'
        libc = ctypes.cdll.LoadLibrary(args.ema_so)
        opts = alignment_options()
        if args.grad_compare:
            grad_compare(model,
                         opts,
                         optimizer,
                         center,
                         teacher_temp_factor,
                         img1, img2,
                         ema_factor,
                         os.path.join(args.output, args.grad),
                         dir_path,
                         grad_count)
        else:
            ipu_model = poptorch.trainingModel(
                model, opts, optimizer=optimizer)
            for i in range(count):
                logits, loss = ipu_model(
                    img1, img2, ema_factor, global_center, teacher_temp_factor)
                print(loss)
                torch.save(logits, f'{dir_path}/logits_{i}.pth')
                torch.save(ipu_model.state_dict(), f'{dir_path}/model{i}.pth')

    else:
        if args.grad_compare:
            for i in range(grad_count):
                grad_dict = {}
                optimizer.zero_grad()
                logits, loss = model(
                    img1, img2, ema_factor, center, teacher_temp_factor)
                loss.backward()
                for n, v in model.named_parameters():
                    grad_dict[f'Gradient___model.{n}'] = v.grad
                optimizer.step()
                model.gpu_update_teacher()
                torch.save(grad_dict, f'{dir_path}/cpu_grad{i}.pt')
                torch.save(model.state_dict(), f'{dir_path}/cpu{i}.pt')
        else:
            for i in range(count):
                optimizer.zero_grad()
                logits, loss = model(
                    img1, img2, ema_factor, center, teacher_temp_factor)
                print(loss)
                loss.backward()
                optimizer.step()
                model.gpu_update_teacher()
                torch.save(logits, f'{dir_path}/logits_{i}.pth')
                torch.save(model.state_dict(), f'{dir_path}/model{i}.pth')


def grad_compare(
        model,
        opts,
        optimizer,
        center,
        teacher_temp_factor,
        img1,
        img2,
        ema_factor,
        path,
        dir_path,
        steps):
    name_list = torch.load(path)
    grad_list = []
    for i, name in enumerate(name_list):
        if 'Gradient___model.' in name or 'UpdatedVar___model.' in name:
            print(name)
            opts.anchorTensor(name, name)
            grad_list.append(name)
    ipu_model = poptorch.trainingModel(model, opts, optimizer=optimizer)
    for i in range(steps):
        _, loss = ipu_model(img1, img2, ema_factor,
                            center, teacher_temp_factor)
        grad_dict = {}
        for name in grad_list:
            grad_ipu = ipu_model.getAnchoredTensor(name)
            grad_dict[name] = grad_ipu
        torch.save(grad_dict, f'{dir_path}/ipu_grad{i}.pt')
        torch.save(ipu_model.state_dict(), f'{dir_path}/ipu{i}.pt')


def pipeline_compare(args, model, optimizer, center):
    img_name = os.path.join(args.output, 'image_pipeline.pth')
    ga = args.ga
    if os.path.exists(img_name):
        img1, img2 = torch.load(img_name)
        print('load image')
    else:
        bs = args.batch_size
        img1 = torch.randint(0, 255, (bs, 2, 3, 224, 224), dtype=torch.uint8)
        img2 = torch.randint(0, 255, (bs, 8, 3, 96, 96), dtype=torch.uint8)
        torch.save([img1, img2], img_name)

    dir_path = os.path.join(args.output, args.device)
    os.makedirs(dir_path, exist_ok=True)

    count = 5
    ema_factor_base = torch.ones((1))
    ema_factor = ema_factor_base * args.momentum_teacher
    global_center = center.repeat(args.ga, 1)
    teacher_temp_factor = 0.04 * torch.ones((args.ga))
    if args.device == 'ipu':
        img1 = torch.cat([img1 for _ in range(ga)])
        img2 = torch.cat([img2 for _ in range(ga)])
        ema_factor = torch.cat([ema_factor for _ in range(ga)])
        assert os.path.exists(args.ema_so), 'please compile custom op ema'
        libc = ctypes.cdll.LoadLibrary(args.ema_so)
        opts = get_options(ga)

        ipu_model = poptorch.trainingModel(model, opts, optimizer=optimizer)
        for i in range(count):
            logits, loss = ipu_model(
                img1, img2, ema_factor, global_center, teacher_temp_factor)
            print(loss)
            torch.save(
                ipu_model.state_dict(),
                f'{dir_path}/pipeline_model{i}.pth')

    else:
        for i in range(count):
            optimizer.zero_grad()
            losses = []
            for j in range(ga):
                logits, loss = model(
                    img1, img2, ema_factor, center, teacher_temp_factor)
                losses.append(loss.item())
                loss.backward()
            optimizer.step()
            model.gpu_update_teacher()
            print(losses)
            torch.save(model.state_dict(), f'{dir_path}/pipeline_model{i}.pth')


def main(args):
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    model = MultiCropWrapper(
        student,
        teacher,
        DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
            act_layer=ERF_GELU
        ),
        DINOHead(
            embed_dim,
            args.out_dim,
            args.use_bn_in_head,
            act_layer=ERF_GELU
        ),
        DINOLoss(
            # total number of crops = 2 global crops + local_crops_number
            args.local_crops_number + 2
        ),
        args.momentum_teacher,
        device=args.device,
        pipeline=args.pipeline,
        alignment=args.alignment)

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(model)

    # ============ building student and teacher networks ... ============
    optimizer = AdamW(params_groups,
                      lr=1.,
                      eps=1e-5)

    optimizer.param_groups[2]['lr'] = 0.
    optimizer.param_groups[3]['lr'] = 0.
    state_name = os.path.join(args.output, 'model.pth')
    if os.path.exists(state_name):
        load_weight(model, state_name)
        print('load_weights')
    else:
        torch.save(model.state_dict(), state_name)

    model.train()
    center = torch.zeros(1, args.out_dim)
    if args.extract_name:
        extract_name(args, model, optimizer, center)
        return

    if args.alignment_pipeline:
        print('alignment pipeline')
        pipeline_compare(args, model, optimizer, center)
    else:
        print('alignment shard')
        shard_alignment(args, model, optimizer, center)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
