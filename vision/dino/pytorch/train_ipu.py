# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright 2022 Facebook, Inc. and its affiliates.
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
import argparse
import os
import sys
import datetime
import time
import timeit
import math
import json
from pathlib import Path
from functools import partial
import numpy as np
import yaml
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import wandb

from core import utils
from core import vision_transformer as vits
from core.dino import DINOLoss, DINOHead, MultiCropWrapper
from core.gelu import ERF_GELU
from core.dataset import DataAugmentationDINO, CustomImageFolder, SynthImageFolder
from core.utils import AverageMeter, save_checkpoint, load_checkpoint, sync_metrics, Precision

import ctypes
import poptorch
from poptorch.optim import SGD, AdamW
from options import train_options
import popdist

config_file = os.path.join(os.path.dirname(__file__), "configs.yml")


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    parser.add_argument(
        '--config',
        type=str,
        help='Configuration Name',
        default='vit_base')
    pargs, remaining_args = parser.parse_known_args()
    config_name = pargs.config
    # Model parameters
    parser.add_argument(
        '--arch',
        default='vit_base',
        type=str,
        choices=[
            'vit_tiny',
            'vit_small',
            'vit_base'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument(
        '--gelu_type',
        default='erf',
        help="gelu type could be erf or tanh")
    parser.add_argument(
        '--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument(
        '--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument(
        '--norm_last_layer',
        default=True,
        type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this parameter to False with vit_small and True with vit_base.""")
    parser.add_argument(
        '--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument(
        '--use_bn_in_head',
        default=False,
        type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--center_momentum', default=0.9, type=float)

    # Temperature teacher parameters
    parser.add_argument(
        '--warmup_teacher_temp',
        default=0.04,
        type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument(
        '--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument(
        '--warmup_teacher_temp_epochs',
        default=0,
        type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.04,
        help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument(
        '--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument(
        '--clip_grad', type=float, default=0., help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch-size')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs of training.')
    parser.add_argument(
        '--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument(
        "--lr", default=0.001, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument(
        "--warmup_epochs",
        default=10,
        type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument(
        '--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument(
        '--optimizer',
        default='adamw',
        type=str,
        choices=[
            'adamw',
            'sgd'],
        help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument(
        '--drop_path_rate',
        type=float,
        default=0.1,
        help="stochastic depth rate")
    parser.add_argument('--eps', type=float, default=1e-8)

    # Multi-crop parameters
    parser.add_argument(
        '--global_crops_scale',
        type=float,
        nargs='+',
        default=(
            0.4,
            1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument(
        '--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument(
        '--local_crops_scale',
        type=float,
        nargs='+',
        default=(
            0.05,
            0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument(
        '--data_path',
        default='',
        type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument(
        '--synthetic_data',
        action="store_true",
        help='Whether or not to use synthethic data. Overrides --data_path.')
    parser.add_argument(
        '--output',
        default="./output",
        type=str,
        help='Path to save checkpoints.')
    parser.add_argument(
        '--log',
        default="loss.log",
        type=str,
        help='Path to save logs.')
    parser.add_argument(
        '--saveckp_freq',
        default=1,
        type=int,
        help='Save checkpoint every x epochs.')
    parser.add_argument(
        '--set_freq',
        default=1,
        type=int,
        help='Set optimizer frequency.')
    parser.add_argument(
        '--print_freq',
        default=10,
        type=int,
        help='Save log every x steps.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument(
        '--threads',
        default=8,
        type=int,
        help='torch number of threads')
    parser.add_argument(
        '--num_workers',
        default=32,
        type=int,
        help='Number of data loading workers.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth')
    parser.add_argument("--wandb", action="store_true", help="Enable Weights and Biases logging")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights and Biases run name")
    
    # IPU
    parser.add_argument('--pipeline', type=int, nargs='+',
                        help='set modules on multi ipus')
    parser.add_argument('--replica', type=int, default=1, help="replica count")
    parser.add_argument(
        '--rts',
        type=bool,
        default=True,
        help="whether use rts")
    parser.add_argument(
        '--ipu_per_replica',
        type=int,
        default=8,
        help="ipu count for one model")
    parser.add_argument(
        '--ga',
        type=int,
        default=16,
        help='gradient accumulation number')
    parser.add_argument(
        '--di',
        type=int,
        default=1,
        help='device iterations number')
    parser.add_argument(
        '--precision',
        type=str,
        default='float32',
        help="precision type for train")
    parser.add_argument(
        '--output_mode',
        default='all',
        help='output mode is final or all')
    parser.add_argument(
        '--async_type',
        default='normal',
        type=str,
        choices=[
            'async',
            'rebatch',
            'normal'],
        help="use async data loader")
    parser.add_argument(
        '--rebatched_worker_size',
        type=int,
        default=128,
        help='rebatched worker size')
    parser.add_argument(
        '--device',
        type=str,
        default='ipu',
        help='device to use')
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=128,
        help='Factor by which to scale the loss and hence gradients to assist numerical stability when using float16.')
    parser.add_argument(
        '--ema_so',
        type=str,
        default='./ema/build/exp_avg_custom_op.so',
        help='custom ema, path of so')
    parser.add_argument(
        "--executable-cache-dir",
        type=str,
        default="./cachedir",
        help = "Directory to cache compiled executables"
    )

    # Load the yaml
    yaml_args = dict()
    if config_name is not None:
        with open(config_file, "r") as f:
            try:
                yaml_args.update(**yaml.safe_load(f)[config_name])
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)

    # Check the yaml args are valid
    known_args = set(vars(parser.parse_args("")))
    unknown_args = set(yaml_args) - known_args
    if unknown_args:
        print(f" Warning: Unknown arg(s) in config file: {unknown_args}")

    parser.set_defaults(**yaml_args)
    args = parser.parse_args()
    return args


def train_dino(args):
    assert os.path.exists(args.ema_so), 'please compile custom op ema'
    ctypes.cdll.LoadLibrary(args.ema_so)
    assert os.path.exists(args.data_path), f'{args.data_path} not exists.'
    if not args.synthetic_data:
        assert os.path.exists(args.data_path), f'{args.data_path} not exists.'

    opts = train_options(
        args.use_popdist,
        args.ipu_per_replica,
        args.pipeline,
        args.ga,
        args.replica,
        args.di,
        args.synthetic_data,
        precision=args.precision,
        output_mode=args.output_mode,
        use_rts=args.rts,
        cachedir=args.executable_cache_dir)

    # ============ preparing data ... ============
    if args.synthetic_data:
        dataset = SynthImageFolder()
    else:
        dataset = CustomImageFolder(
            args.data_path,
            transform=DataAugmentationDINO(
                args.global_crops_scale,
                args.local_crops_scale,
                args.local_crops_number)
        )
    if args.async_type == 'async':
        mode = poptorch.DataLoaderMode.Async
        data_loader = poptorch.DataLoader(
            options=opts,
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
            worker_init_fn=None,
            mode=mode,
            async_options={'load_indefinitely': True},
        )
    elif args.async_type == 'rebatch':
        mode = poptorch.DataLoaderMode.AsyncRebatched
        data_loader = poptorch.DataLoader(
            options=opts,
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
            worker_init_fn=None,
            mode=mode,
            async_options={'load_indefinitely': True},
            rebatched_worker_size=args.rebatched_worker_size,
        )

    else:
        data_loader = poptorch.DataLoader(
            options=opts,
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=True,
        )

    print(f"Number of images loaded: {len(dataset)}. Steps in data loader: {len(data_loader)}")
    # ============ building student and teacher networks ... ============
    if args.gelu_type == 'erf':
        gelu = partial(ERF_GELU, precision=args.precision)
    else:
        gelu = nn.GELU
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
            act_layer=gelu,
            precision=args.precision,
        )
        teacher = vits.__dict__[
            args.arch](
            patch_size=args.patch_size,
            act_layer=gelu,
            precision=args.precision)
        embed_dim = student.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    mt_schedule = utils.cosine_scheduler(
        args.momentum_teacher,
        1.0,
        args.epochs, len(data_loader)
    )
    teacher_temp_schedule = np.concatenate(
        (np.linspace(
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs),
            np.ones(
            args.epochs -
            args.warmup_teacher_temp_epochs) *
            args.teacher_temp))

    # multi-crop wrapper handles forward with inputs of different resolutions
    model = MultiCropWrapper(
        student,
        teacher,
        DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
            act_layer=gelu,
            precision=args.precision,
        ),
        DINOHead(
            embed_dim,
            args.out_dim,
            args.use_bn_in_head,
            act_layer=gelu,
            precision=args.precision,
        ),
        DINOLoss(
            # total number of crops = 2 global crops + local_crops_number
            args.local_crops_number + 2,
        ),
        args.momentum_teacher,
        device=args.device,
        pipeline=args.pipeline,
        precision=args.precision)

    print(f'train on ipu with {args.precision}.')
    if args.precision is Precision.FP16:
        model.half()
    model.train()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(model)
    print(
        f'use optimizer : {args.optimizer}, clip_grad={args.clip_grad}, eps={args.eps}')
    use_half = False
    if args.precision is not Precision.FP32:
        use_half = True
    if args.optimizer == "adamw":
        print(f'optimizer max_grad_norm = {args.clip_grad}')
        optimizer = AdamW(params_groups,
                          lr=args.min_lr,
                          betas=(0.9, 0.999),
                          eps=args.eps,
                          weight_decay=args.weight_decay,
                          # nonzero to make wd a variable in the optimizer
                          loss_scaling=args.loss_scale if use_half else None,
                          max_grad_norm=args.clip_grad if args.clip_grad > 1e-5 else None)

    elif args.optimizer == "sgd":
        optimizer = SGD(params_groups,
                        lr=args.min_lr,
                        momentum=0.9,
                        weight_decay=args.weight_decay,
                        # nonzero to make wd a variable in the optimizer
                        loss_scaling=args.loss_scale if use_half else None)
    start_epoch = 0
    center = torch.zeros(1, args.out_dim)
    if args.resume:
        model_path = os.path.join(args.output, args.checkpoint)
        start_epoch, center = load_checkpoint(model, optimizer, model_path)
        start_epoch += 1
        print(f'load {model_path} success, train start at epoch:{start_epoch}')
    ipu_model = poptorch.trainingModel(model, opts, optimizer=optimizer)

    print("Starting DINO training !")
    log_path = os.path.join(args.output, args.log)
    for epoch in range(start_epoch, args.epochs):
        end = time.time()
        center = train_one_epoch(ipu_model,
                                 optimizer,
                                 data_loader,
                                 epoch,
                                 lr_schedule,
                                 wd_schedule,
                                 mt_schedule,
                                 teacher_temp_schedule[epoch],
                                 args.freeze_last_layer,
                                 end,
                                 log_path,
                                 center,
                                 args,)
        save_checkpoint(
            epoch,
            ipu_model,
            optimizer,
            center,
            args.output,
            epoch %
            args.saveckp_freq == 0)


def train_one_epoch(model,
                    optimizer,
                    data_loader,
                    epoch,
                    lr_schedule,
                    wd_schedule,
                    mt_schedule,
                    teacher_temp_epoch,
                    freeze_last_layer,
                    end,
                    log_path,
                    center,
                    args,):
    losses = AverageMeter('loss', ':.2f')
    batch_time = AverageMeter('batch', ':.2f')
    data_time = AverageMeter('data', ':.2f')
    throughput = AverageMeter('throughput', ':.0f')
    steps_per_epoch = len(data_loader)
    # batch dim must be multiple of device_iterations * replication_factor *
    # gradient_accumulation
    global_count = args.di * args.replica * args.ga
    ema_factor_base = torch.ones((global_count))
    ema_factor = ema_factor_base * 1
    teacher_temp_factor = teacher_temp_epoch * torch.ones((global_count))
    for it, (input_224, input_96) in enumerate(data_loader):
        start_time = timeit.default_timer()

        data_time.update(time.time() - end)
        current_step = it + epoch * steps_per_epoch
        if it % args.set_freq == 0:
            lr = lr_schedule[current_step]
            lr = max(lr, args.min_lr)
            wd = wd_schedule[current_step]
            mt = mt_schedule[current_step]
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[0]['weight_decay'] = wd
            optimizer.param_groups[1]['lr'] = lr
            # set lr_schedule value for last layer separately
            if epoch >= freeze_last_layer:
                # update last_layer weight_v, not update weight_g
                optimizer.param_groups[3]['lr'] = lr
                optimizer.param_groups[3]['weight_decay'] = wd
                if not args.norm_last_layer:
                    optimizer.param_groups[2]['lr'] = lr
                    optimizer.param_groups[2]['weight_decay'] = wd
            model.setOptimizer(optimizer)  # apply changes

        ema_factor = ema_factor_base * mt
        global_center = center.repeat(global_count, 1)

        batch_center, loss = model(
            input_224, input_96, ema_factor, global_center, teacher_temp_factor)

        batch_center = torch.mean(batch_center, dim=0, keepdim=True)
        center = center * args.center_momentum + \
            batch_center * (1. - args.center_momentum)

        end_time = timeit.default_timer()
        tput = input_224.size(0) / (end_time - start_time)

        loss = sync_metrics(torch.mean(loss))
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss, input_224.size(0))
        tput = input_224.size(0) / batch_time.val

        last_layer_lr = optimizer.param_groups[3]['lr']
        throughput.update(sync_metrics(tput, False))
        info = (f'[{epoch}/{args.epochs}|{it}/{steps_per_epoch}]\t'
                f'lr:{lr:.3e} - \t'
                f'last_layer_lr:{last_layer_lr:.3e} - \t'
                f'wd:{wd:.3e} - \t'
                f'mt:{mt:.4e} - \t'
                f'{losses} - \t'
                f'{batch_time} - \t'
                f'{data_time} - \t'
                f'{throughput} samples/sec\n')

        with open(log_path, 'a') as fw:
            fw.write(info)
            if it % args.print_freq == 0:
                print(info)
                if args.wandb and (not args.use_popdist or args.popdist_rank == 0):
                    # if wandb has not been initialised
                    if wandb.run is None:
                        wandb.init(
                           config=vars(args),
                            project="torch-dino",
                            name=args.wandb_run_name
                        )
                    wandb.log({
                        "loss": losses.val,
                        "batch_time": batch_time.val,
                        "data_time": data_time.val,
                        "throughput": throughput.val,
                    })
    return center


if __name__ == '__main__':
    args = get_args_parser()
    torch.set_num_threads(args.threads)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    dump_config_name = os.path.join(args.output, f'pretrain_{args.arch}.yaml')
    with open(dump_config_name, 'w') as fw:
        yaml.safe_dump(args.__dict__, fw)
    args.precision = Precision(args.precision)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    utils.init_popdist(args)
    train_dino(args)
