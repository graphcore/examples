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

import time
import numpy as np
import os
from pathlib import Path
import torch
import poptorch
import yaml
from util.ipu_mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from util.datasets import build_dataset
import core.models_vit as models_vit
from options import finetune_options
from core import utils
from util.log import AverageMeter, ProgressMeter, logger, WandbLog
from timm.utils import accuracy
import logging
from argparser import get_args_parser
import wandb


@torch.no_grad()
def evaluate(data_loader, model, print_freq):
    criterion = torch.nn.CrossEntropyLoss()

    batch_time = AverageMeter('time', ':6.3f')
    data_time = AverageMeter('data', ':6.3f')
    losses = AverageMeter('loss', ':.3f')
    tput = AverageMeter('throughput', ':.0f')
    acc1_log = AverageMeter('acc1', ':.3f')
    acc5_log = AverageMeter('acc5', ':.3f')
    # switch to evaluation mode
    model.eval()
    meters = [batch_time, data_time, losses, tput, acc1_log, acc5_log]
    progress = ProgressMeter(
        len(data_loader),
        meters,
        prefix="Evluation: "
    )

    if args.wandb:
        wandb_logger = WandbLog(meters)

    end = time.time()
    for it, batch in enumerate(data_loader):
        data_time.update(time.time() - end)
        print(data_time.val)
        images = batch[0]
        target = batch[-1]
        # compute output
        output = model(images)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]

        acc1 = torch.mean(acc1)
        acc5 = torch.mean(acc5)

        acc1_log.update(acc1.item(), batch_size)
        acc5_log.update(acc5.item(), batch_size)
        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - end)

        end = time.time()

        tput.update(utils.sync_metrics(batch_size / batch_time.val))

        if it % print_freq == 0:
            if not args.use_popdist or (
                    args.use_popdist and args.popdist_rank == 0):
                logger.info(progress.display(it))
                if args.wandb:
                    wandb_logger.log()

    logger.info('* Acc@1 {top1} Acc@5 {top5} loss {losses}'
                .format(top1=acc1_log, top5=acc5_log, losses=losses))

    return {meter.name: meter.avg for meter in meters}


def load_eval_checkpoint(model, path):
    assert os.path.exists(path), f'{path} not exists'
    model_state = torch.load(path)
    weights = model_state['model']

    model.load_state_dict(weights)
    logger.info(f"Loaded checkpoint from path: {path}")
    return model


def main(args):
    log_path = os.path.join(args.output, 'eval')
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    fileHandler = logging.FileHandler(
        log_path + '_' + now + '.log',
        mode='w',
        encoding='UTF-8')
    fileHandler.setLevel(logging.NOTSET)
    logger.addHandler(fileHandler)
    args.async_type = 'async'
    args.replica = 1
    args.batch_size = 16
    args.di = 1000
    opts_infer = finetune_options(
        gradient_accumulation_count=args.gradient_accumulation_count,
        replica=args.replica,
        half=args.half,
        ipu_per_replica=args.ipus,
        device_iterations=args.device_iterations,
        opt_type='eval')

    logger.info(
        'job dir: {}'.format(
            os.path.dirname(
                os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(', ', ',\n'))
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_val = build_dataset(is_train=False, args=args)

    if args.async_type == 'async':
        mode = poptorch.DataLoaderMode.Async
        data_loader_val = poptorch.DataLoader(
            options=opts_infer,
            dataset=dataset_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=True,
            persistent_workers=True,
            worker_init_fn=None,
            mode=mode,
            async_options={'load_indefinitely': True},
        )
    elif args.async_type == 'rebatch':
        mode = poptorch.DataLoaderMode.AsyncRebatched

        data_loader_val = poptorch.DataLoader(
            options=opts_infer,
            dataset=dataset_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=True,
            persistent_workers=True,
            worker_init_fn=None,
            mode=mode,
            async_options={'load_indefinitely': True},
            rebatched_worker_size=args.rebatched_worker_size,
        )
    else:
        data_loader_val = poptorch.DataLoader(
            options=opts_infer,
            dataset=dataset_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True,
        )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        logger.info("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes)

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model = models_vit.__dict__[args.model](
        criterion=criterion,
        pipeline=[3, 3, 3, 3],
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info('number of params (M): %.2f' % (n_parameters / 1.e6))

    model = load_eval_checkpoint(model, args.resume)

    # switch to evaluation mode before creating an inference model
    model.eval()
    ipu_infer_model = poptorch.inferenceModel(model, options=opts_infer)
    test_stats = evaluate(
        data_loader_val,
        ipu_infer_model,
        args.print_freq)
    logger.info(f"Accuracy of the network on the {len(data_loader_val)} test images: {test_stats['acc1']:.1f}%")


if __name__ == '__main__':
    args = get_args_parser()

    utils.init_popdist(args)

    if args.wandb:
        if not args.use_popdist or (
                args.use_popdist and args.popdist_rank == 0):
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

    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
    main(args)
