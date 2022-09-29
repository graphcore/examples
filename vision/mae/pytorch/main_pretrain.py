# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
import sys
import time
import argparse
from pathlib import Path
import json
import yaml
import numpy as np
import torch
import torchvision.transforms as transforms
import timm
import timm.optim.optim_factory as optim_factory
import poptorch
from poptorch.optim import AdamW
import wandb

import util.lr_sched as lr_sched
from util.log import AverageMeter, ProgressMeter, WandbLog, logger
from util.checkpoint import save_checkpoint, load_checkpoint
from options import train_options
from core import models_mae
from core import utils
import ctypes
from core.utils import sync_metrics
import logging
from util.datasets import GeneratedData, ImageFolder, get_compile_datum
threads = 4
os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
os.environ["MKL_NUM_THREADS"] = str(threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
torch.set_num_threads(threads)
config_file = os.path.join(os.path.dirname(__file__), "configs.yml")


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument(
        '--config',
        default='vit_base',
        type=str,
        help='Configuration name')
    pargs, remaining_args = parser.parse_known_args()
    config_name = pargs.config
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)

    # Model parameters
    parser.add_argument(
        '--model',
        default='mae_vit_base_patch16',
        type=str,
        metavar='MODEL',
        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument(
        '--norm_pix_loss',
        action='store_true',
        help='Use (per-patch) normalized pixels as targets for computing loss')

    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--sequence_length', type=int, default=196)
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument(
        '--blr',
        type=float,
        default=1e-3,
        metavar='LR',
        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument(
        '--data_path',
        default='/datasets01/imagenet_full_size/061417/',
        type=str,
        help='dataset path')

    parser.add_argument(
        '--generated_data', action='store_true', help='Use host generated data instead of real imagenet data.'
    )

    parser.add_argument('--output', default='./mae',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log', default='log_info.txt',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='ipu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from checkpoint')

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--print_freq', default=10)
    parser.add_argument('--saveckp_freq', default=10)
    parser.add_argument('--optimizer_step', default=10)

    # IPU related
    parser.add_argument('--pipeline', type=int, nargs='+',
                        help='set modules on multi ipus')
    parser.add_argument(
        '--remap_so',
        type=str,
        default='./remap/remap_ops.so',
        help='custom remap, path of so')
    parser.add_argument(
        '--gradient_accumulation_count',
        default=128,
        type=int,
        help='gradient accumulate')
    parser.add_argument(
        '--replica',
        default=1,
        type=int,
        help='model replic count')
    parser.add_argument('--ipus', default=4, type=int,
                        help='ipu count for one model')
    parser.add_argument('--half', action='store_true', help='if use float16')
    parser.add_argument('--rts', action='store_true', help='if use rts')
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
    parser.add_argument('--loss_scale', type=float, default=128.0)


    # WandB related
    parser.add_argument('--wandb', action='store_true',
                        help='Turn on Weights and Biases logging.')
    parser.add_argument('--wandb_project_name', default='torch-mae',
                        type=str, help='Weights and Biases project name.')
    parser.add_argument('--wandb_run_name', default=None,
                        type=str, help='Weights and Biases run name.')

    # compile only
    parser.add_argument('--compile_only', action='store_true',
                        help='Exit after compiling model.')
    yaml_args = dict()
    if config_name is not None:
        with open(config_file, 'r') as f:
            try:
                yaml_args.update(**yaml.safe_load(f)[config_name])
            except yaml.YAMLError as exc:
                logger.info(exc)
                sys.exit()
    # check the yaml args are valid
    known_args = set(vars(parser.parse_args('')))
    unknown_args = set(yaml_args) - known_args
    if unknown_args:
        logger.info(f" Warning: Unknown arg(s) in config file: {unknown_args}")
    parser.set_defaults(**yaml_args)
    args = parser.parse_args()
    # helper args
    args.pretrain = True
    return args


class To_Tensor(torch.nn.Module):
    def forward(self, img):
        out = torch.from_numpy(np.array(img))
        out = out.permute(2, 0, 1)
        return out


def main(args):
    log_path = os.path.join(args.output, args.log)
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    fileHandler = logging.FileHandler(
        log_path + '_' + now + '.log',
        mode='w',
        encoding='UTF-8')
    fileHandler.setLevel(logging.NOTSET)
    logger.addHandler(fileHandler)
    assert os.path.exists(args.remap_so), 'please compile custom op remap'
    ctypes.cdll.LoadLibrary(args.remap_so)
    opts = train_options(
        args.use_popdist,
        gradient_accumulation_count=args.gradient_accumulation_count,
        replica=args.replica,
        half=args.half,
        als=args.use_als,
        ipu_per_replica=args.ipus,
        rts=args.rts)
    # simple augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(
            args.input_size, scale=(
                0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        To_Tensor()
    ])


    if not args.generated_data:
        dataset_train = ImageFolder(
            os.path.join(args.data_path, 'train'), transform=transform_train, use_half=args.half)
    else:
        dataset_train = GeneratedData(args.input_size, args.half, image_transform=transform_train, pretrain=args.pretrain)


    if args.async_type == 'async':
        mode = poptorch.DataLoaderMode.Async
        data_loader_train = poptorch.DataLoader(
            options=opts,
            dataset=dataset_train,
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
        data_loader_train = poptorch.DataLoader(
            options=opts,
            dataset=dataset_train,
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
        data_loader_train = poptorch.DataLoader(
            options=opts,
            dataset=dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )

    # define the model
    model = models_mae.__dict__[
        args.model](
        norm_pix_loss=args.norm_pix_loss,
        pipeline=args.pipeline,
        device=args.device,
        mask_ratio=args.mask_ratio,
        gelu_type=args.gelu_type,
        half=args.half)

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.local_batch_size / 256
    logger.info("base lr: %.2e" % (args.lr * 256 / args.local_batch_size))
    logger.info("actual lr: %.2e" % args.lr)
    logger.info("effective batch size: %d" % args.local_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    if args.half:
        optimizer = AdamW(
            param_groups, lr=args.lr, betas=(
                0.9, 0.95), accum_type=torch.float16, loss_scaling=args.loss_scale)
        model.half()
    else:
        optimizer = AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    logger.info(optimizer)

    start_epoch = 0
    if args.resume:
        model_path = os.path.join(args.output, args.checkpoint)
        start_epoch = load_checkpoint(model, optimizer, model_path)
        start_epoch += 1
        logger.info(f'load {model_path} success, train start at epoch:{start_epoch}')
    logger.info(f"Start training for {args.epochs} epochs")

    model.train()
    ipu_model = poptorch.trainingModel(
        model, options=opts, optimizer=optimizer)

    start_train = time.perf_counter()
    logger.info("Compiling..")

    samples, ids_shuffle, ids_restore, keep_mat, restore_mat, mask = get_compile_datum(args, opts, dataset_train)
    ipu_model.compile(samples, samples, ids_restore,
                      keep_mat, restore_mat, mask)
    end_compile = time.perf_counter()
    compile_time = end_compile - start_train
    logger.info(f"Compilation time: {compile_time:.3f} secs")

    if args.compile_only:
        sys.exit(0)

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(
            ipu_model,
            data_loader_train,
            optimizer,
            epoch,
            args
        )
        if epoch % args.saveckp_freq == 0 or epoch == (
                args.epochs - 1) or epoch == 100:
            save_checkpoint(epoch, ipu_model, optimizer, args.output)
    stop_train = time.perf_counter()
    duration_run = stop_train - start_train
    logger.info(f"Training time: {duration_run:.3f} secs")


def train_one_epoch(model,
                    data_loader,
                    optimizer,
                    epoch,
                    args):
    batch_time = AverageMeter('time', ':6.3f')
    data_time = AverageMeter('data', ':6.3f')
    losses = AverageMeter('loss', ':.4e')
    tput = AverageMeter('throughput', ':.0f')
    lres = AverageMeter('LR', ':.6f')
    meters = [batch_time, data_time, losses, tput, lres]
    progress = ProgressMeter(
        len(data_loader),
        meters,
        prefix="Training Epoch: [{}]".format(epoch))
    if args.wandb:
        wandb_logger = WandbLog(meters)

    end = time.time()
    for data_iter_step, (samples, ids_shuffle, ids_restore,
                         keep_mat, restore_mat, mask) in enumerate(data_loader):

        data_time.update(time.time() - end)
        _, loss = model(samples, samples, ids_restore,
                        keep_mat, restore_mat, mask)
        loss = sync_metrics(torch.mean(loss))
        lr_sched.adjust_learning_rate(
            optimizer,
            data_iter_step /
            len(data_loader) +
            epoch,
            args)

        if data_iter_step % args.optimizer_step == 0:
            model.setOptimizer(optimizer)
        losses.update(loss, samples.size(0))
        lr = optimizer.param_groups[0]["lr"]
        lres.update(lr)
        batch_time.update(time.time() - end)
        end = time.time()

        if args.use_popdist:
            tput.update(sync_metrics(samples.shape[0] / batch_time.val, average=False))
        else:
            tput.update(samples.shape[0] / batch_time.val)

        if data_iter_step % args.print_freq == 0:
            if not args.use_popdist or (args.use_popdist and args.popdist_rank == 0):
                log_message = progress.display(data_iter_step)
                if data_iter_step % args.print_freq == 0:
                    logger.info(log_message)
                    logger.info(samples.shape)
                    if args.wandb:
                        wandb_logger.log()


if __name__ == '__main__':
    args = get_args_parser()
    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
    dump_config_name = os.path.join(args.output, f'pretrain_{args.model}.yaml')
    with open(dump_config_name, 'w') as fw:
        yaml.safe_dump(args.__dict__, fw)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    utils.init_popdist(args)

    if args.wandb:
        if not args.use_popdist or (args.use_popdist and args.popdist_rank == 0):
            wandb.init(
                project=args.wandb_project_name,
                name=args.wandb_run_name,
                settings=wandb.Settings(console="wrap"),
                config=vars(args)
            )
            if args.half:
                wandb.config.update({"precision": "16.16"})
            else:
                wandb.config.update({"precision": "32.32"})
    main(args)
