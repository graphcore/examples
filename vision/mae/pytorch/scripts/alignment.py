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
from util.log import logger
from core import utils
from core import models_mae
import timm.optim.optim_factory as optim_factory
from options import alignment_options
from core.gelu import ERF_GELU


def get_args_parser():
    parser = argparse.ArgumentParser('MAE', add_help=False)

    # Model parameters
    parser.add_argument('--batch_size', default=1, type=int, help='batch-size')

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
    parser.add_argument('--output', type=str, default='./alignment')
    parser.add_argument('--grad', type=str, default='grad_names.pt')
    parser.add_argument('--alignment_pipeline', action='store_true')
    parser.add_argument('--gradient_accumulation_count', default=12, type=int)
    parser.add_argument(
        '--remap_so',
        type=str,
        default='../remap/remap_ops.so',
        help='custom remap, path of so')
    return parser


def get_data(bs):
    mask_ratio = 0.75
    img = torch.randint(0, 255, (bs, 3, 224, 224), dtype=torch.uint8)
    N, L, D = bs, 196, 768
    noise = torch.rand(N, L)  # noise in [0, 1]
    # sort noise for each sample
    # ascend: small is keep, large is remove
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    len_keep = int(L * (1 - mask_ratio))
    keep_mat = torch.zeros(N, L, len_keep)
    ids_keep = ids_shuffle[:, :len_keep]
    keep_mat.scatter_(dim=1, index=ids_keep.view(N, 1, len_keep), value=1)
    restore_mat = torch.zeros(N, L, L)
    restore_mat.scatter_(dim=1, index=ids_restore.view(N, 1, L), value=1)

    mask = torch.ones([N, L])
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return img, noise, ids_shuffle, ids_restore, keep_mat, restore_mat, mask


def extract_name(args, model, optimizer):
    ctypes.cdll.LoadLibrary(args.remap_so)
    path = os.path.join(args.output, args.grad)
    opts = alignment_options()
    ipu_model = poptorch.trainingModel(model, opts, optimizer=optimizer)
    input_data, noise, ids_shuffle, ids_restore, keep_mat, restore_mat, mask = get_data(
        bs=1)
    _, loss = ipu_model(input_data, input_data, ids_restore,
                        keep_mat, restore_mat, mask)
    tensor_names = ipu_model.getTensorNames()
    name_cpu = [
        'Gradient___model.' +
        name for name,
        parms in model.named_parameters()]
    names = list(set(tensor_names).intersection(set(name_cpu)))
    torch.save(names, path)


def load_weight(model, path):
    state_dict = torch.load(path)
    new_dict = OrderedDict()
    for n, v in model.state_dict().items():
        if n in state_dict:
            new_dict[n] = state_dict[n]
        else:
            logger.info(f'{n} not in state dict')
            new_dict[n] = v
    model.load_state_dict(new_dict)


def shard_alignment(args, model, optimizer):
    assert os.path.exists(args.remap_so), 'please compile custom op remap'
    img_name = os.path.join(args.output, 'image.pth')
    if os.path.exists(img_name):
        input_data, noise, ids_shuffle, ids_restore, keep_mat, restore_mat, mask = torch.load(
            img_name)
        logger.info(f'load image')
    else:
        bs = args.batch_size
        input_data, noise, ids_shuffle, ids_restore, keep_mat, restore_mat, mask = get_data(
            bs)
        torch.save([input_data, noise, ids_shuffle, ids_restore,
                   keep_mat, restore_mat, mask], img_name)

    compare_weights = False
    dir_path = os.path.join(args.output, args.device)
    os.makedirs(dir_path, exist_ok=True)
    count = 10
    grad_count = 5
    if args.device == 'ipu':
        ctypes.cdll.LoadLibrary(args.remap_so)
        opts = alignment_options()
        if args.grad_compare:
            grad_compare(model,
                         opts,
                         optimizer,
                         [input_data,
                          noise,
                          ids_shuffle,
                          ids_restore,
                          keep_mat,
                          restore_mat,
                          mask],
                         os.path.join(args.output,
                                      args.grad),
                         dir_path,
                         grad_count)
        else:
            ipu_model = poptorch.trainingModel(
                model, opts, optimizer=optimizer)
            for i in range(count):
                logits, loss = ipu_model(
                    input_data, input_data, ids_restore, keep_mat, restore_mat, mask)
                logger.info(f'loss: {loss}')
                torch.save(logits, f'{dir_path}/logits_{i}.pth')
                torch.save(ipu_model.state_dict(), f'{dir_path}/model{i}.pth')

    else:
        if args.grad_compare:
            for i in range(grad_count):
                grad_dict = {}
                optimizer.zero_grad()
                logits, loss = model(
                    input_data, input_data, ids_restore, keep_mat, restore_mat, mask, ids_shuffle=ids_shuffle)
                loss.backward()
                for n, v in model.named_parameters():
                    grad_dict[f'Gradient___model.{n}'] = v.grad
                optimizer.step()
                torch.save(grad_dict, f'{dir_path}/cpu_grad{i}.pt')
                torch.save(model.state_dict(), f'{dir_path}/cpu{i}.pt')
        else:
            for i in range(count):
                optimizer.zero_grad()
                logits, loss = ipu_model(
                    input_data, input_data, ids_restore, keep_mat, restore_mat, mask, ids_shuffle=ids_shuffle)
                logger.info(f'loss: {loss}')
                loss.backward()
                optimizer.step()
                torch.save(logits, f'{dir_path}/logits_{i}.pth')
                torch.save(model.state_dict(), f'{dir_path}/model{i}.pth')


def grad_compare(
        model,
        opts,
        optimizer,
        input,
        path,
        dir_path,
        steps):
    input_data, noise, ids_shuffle, ids_restore, keep_mat, restore_mat, mask = input
    name_list = torch.load(path)
    grad_list = []
    for i, name in enumerate(name_list):
        if 'Gradient___model.' in name or 'UpdatedVar___model.' in name:
            logger.info(f'name: {name}')
            opts.anchorTensor(name, name)
            grad_list.append(name)
    ipu_model = poptorch.trainingModel(model, opts, optimizer=optimizer)
    for i in range(steps):
        _, loss = ipu_model(input_data, input_data,
                            ids_restore, keep_mat, restore_mat, mask)
        grad_dict = {}
        for name in grad_list:
            grad_ipu = ipu_model.getAnchoredTensor(name)
            grad_dict[name] = grad_ipu
        torch.save(grad_dict, f'{dir_path}/ipu_grad{i}.pt')
        torch.save(ipu_model.state_dict(), f'{dir_path}/ipu{i}.pt')


def pipeline_compare(args, model, optimizer, center):
    img_name = os.path.join(args.output, 'image_pipeline.pth')
    gradient_accumulation_count = args.gradient_accumulation_count
    if os.path.exists(img_name):
        input_data, noise, ids_shuffle, ids_restore, keep_mat, restore_mat, mask = torch.load(
            img_name)
        logger.info(f'load image')
    else:
        bs = args.batch_size
        input_data, noise, ids_shuffle, ids_restore, keep_mat, restore_mat, mask = get_data(
            bs)
        torch.save([input_data, noise, ids_shuffle, ids_restore,
                   keep_mat, restore_mat, mask], img_name)

    dir_path = os.path.join(args.output, args.device)
    os.makedirs(dir_path, exist_ok=True)

    count = 5
    if args.device == 'ipu':
        ipu_model = poptorch.trainingModel(model, opts, optimizer=optimizer)
        for i in range(count):
            _, loss = ipu_model(input_data, input_data,
                                ids_restore, keep_mat, restore_mat, mask)
            logger.info(f'loss: {loss}')

            torch.save(
                ipu_model.state_dict(),
                f'{dir_path}/pipeline_model{i}.pth')

    else:
        for i in range(count):
            optimizer.zero_grad()
            losses = []
            for j in range(gradient_accumulation_count):
                _, loss = ipu_model(
                    input_data, input_data, ids_restore, keep_mat, restore_mat, mask, ids_shuffle=ids_shuffle)
                losses.append(loss.item())
                loss.backward()
            optimizer.step()
            logger.info(f'loss: {losses}')
            torch.save(model.state_dict(), f'{dir_path}/pipeline_model{i}.pth')


def main(args):

    # multi-crop wrapper handles forward with inputs of different resolutions
    model = models_mae.__dict__[
        'mae_vit_align_patch16'](
        norm_pix_loss=True,
        pipeline=args.pipeline,
        device=args.device,
        mask_ratio=0.75,
        half=False)

    param_groups = optim_factory.add_weight_decay(model, 0.05)
    optimizer = AdamW(param_groups,
                      lr=0.5,
                      eps=1e-5)

    state_name = os.path.join(args.output, 'model.pth')
    if os.path.exists(state_name):
        load_weight(model, state_name)
        logger.info(f'load image')
    else:
        torch.save(model.state_dict(), state_name)

    model.train()
    if args.extract_name:
        extract_name(args, model, optimizer)
        return

    if args.alignment_pipeline:
        logger.info(f'alignment pipeline')
        pipeline_compare(args, model, optimizer)
    else:
        logger.info(f'alignment shard')
        shard_alignment(args, model, optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MAE', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
