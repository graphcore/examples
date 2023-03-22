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

import logging
import argparse
import datetime

import numpy as np
import os
from pathlib import Path
import poptorch
from poptorch.optim import AdamW
import yaml
from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import util.lr_decay as lrd
from util.datasets import build_dataset, get_compile_datum
from util.pos_embed import interpolate_pos_embed
import core.models_vit as models_vit
from options import finetune_options
from core import utils
import sys
from typing import Iterable, Optional
import torch
from util.ipu_mixup import Mixup
from core.utils import sync_metrics
import util.lr_sched as lr_sched
from util.log import AverageMeter, ProgressMeter, logger, WandbLog
from util.checkpoint import save_checkpoint, load_checkpoint
import time
from core.utils import AverageMeter
from argparser import get_args_parser
import wandb

config_file = os.path.join(os.path.dirname(__file__), "configs.yml")


class collater:
    def __init__(self, args, mixup_fn=None):
        self.mixup_fn = mixup_fn
        self.args = args

    def __call__(self, batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        data = torch.stack(data)
        target = torch.tensor(target)
        if data.shape[0] % 2 == 0:
            data, targets = self.mixup_fn(data, target)
        else:

            logger.info("WARNING: Batchsize is not even! ")
            # fix data shape for incomplete batch when rebatch is enabled
            if data.shape[0] == 1:
                data, targets = self.mixup_fn(data.repeat(2, 1, 1, 1), target.repeat(2))
            else:
                data, targets = self.mixup_fn(data[0:-1, :, :, :], target[0:-1])
        if self.args.half:
            data = data.half()
        return [data, targets]


def main(args):
    log_path = os.path.join(args.output, args.log)
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    fileHandler = logging.FileHandler(log_path + "_" + now + ".log", mode="w", encoding="UTF-8")
    fileHandler.setLevel(logging.NOTSET)
    logger.addHandler(fileHandler)
    opts = finetune_options(
        gradient_accumulation_count=args.gradient_accumulation_count,
        replica=args.replica,
        half=args.half,
        ipu_per_replica=args.ipus,
        device_iterations=args.device_iterations,
    )
    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
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
            num_classes=args.nb_classes,
        )
        collate_fn = collater(args, mixup_fn)
    else:
        collate_fn = None
    mode = poptorch.DataLoaderMode.AsyncRebatched
    data_loader_train = poptorch.DataLoader(
        options=opts,
        dataset=dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        mode=mode,
        async_options={"early_preload": True, "miss_sleep_time_in_ms": 0, "buffer_size": 4},
        persistent_workers=True,
        num_workers=args.num_workers,
        drop_last=True,
        rebatched_worker_size=args.rebatched_worker_size,
        collate_fn=collate_fn,
    )

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:

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
    if args.finetune and not args.resume:
        checkpoint = torch.load(args.finetune, map_location="cpu")

        logger.info(f"Load pre-trained checkpoint from: {args.finetune}")
        checkpoint_model = checkpoint["model"]

        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        logger.info(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}
        else:
            assert set(msg.missing_keys) == {"head.weight", "head.bias"}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.half:
        model.half()
    logger.info("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * args.gradient_accumulation_count * args.replica
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    logger.info("base lr: %.2e" % (args.lr * 256 / args.local_batch_size))
    logger.info("actual lr: %.2e" % args.lr)

    logger.info("accumulate grad iterations: %d" % args.gradient_accumulation_count)
    logger.info("effective batch size: %d" % args.local_batch_size)

    param_groups = lrd.param_groups_lrd(
        model, args.weight_decay, no_weight_decay_list=model.no_weight_decay(), layer_decay=args.layer_decay
    )
    optimizer = AdamW(param_groups, lr=args.lr, loss_scaling=args.loss_scale if args.half else None)
    model = model.train()
    model = poptorch.trainingModel(model, options=opts, optimizer=optimizer)
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1
    logger.info("criterion = %s" % str(criterion))

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    logger.info("Compiling..")
    loader = iter(data_loader_train)
    datum = next(loader)
    samples, targets = get_compile_datum(args, opts, dataset_train, collate_fn)
    model.compile(samples, targets)
    end_compile = time.time()
    compile_time = end_compile - start_time
    logger.info(f"Compilation time: {compile_time:.3f} secs")

    if args.compile_only:
        sys.exit(0)
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, data_loader_train, optimizer, epoch, args=args)
        if epoch % args.saveckp_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(epoch, model, optimizer, args.output)

        if epoch == args.epochs - 1:
            save_checkpoint(epoch, model, optimizer, args.output)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


def train_one_epoch(
    model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, args=None
):
    header = "Epoch: [{}]".format(epoch)

    batch_time = AverageMeter("time", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    losses = AverageMeter("loss", ":.4e")
    tput = AverageMeter("throughput", ":.0f")
    lres = AverageMeter("LR", ":.6f")
    meters = [batch_time, data_time, losses, tput, lres]
    progress = ProgressMeter(
        len(data_loader), [batch_time, data_time, losses, tput, lres], prefix="Training Epoch: [{}]".format(epoch)
    )
    end = time.time()
    if args.wandb:
        wandb_logger = WandbLog(meters)

    end = time.time()
    for data_iter_step, (samples, targets) in enumerate(data_loader):
        data_time.update(time.time() - end)
        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        model.setOptimizer(optimizer)
        outputs, loss = model(samples, targets)
        losses.update(torch.mean(loss).item(), samples.size(0))
        lr = optimizer.param_groups[0]["lr"]
        lres.update(lr)
        batch_time.update(time.time() - end)
        end = time.time()
        tput.update(sync_metrics(samples.shape[0] / batch_time.val))
        log_message = progress.display(data_iter_step)
        if data_iter_step % args.print_freq == 0:
            if not args.use_popdist or (args.use_popdist and args.popdist_rank == 0):
                logger.info(log_message)
                if args.wandb:
                    wandb_logger.log()


if __name__ == "__main__":
    args = get_args_parser()
    utils.init_popdist(args)
    if args.wandb:
        if not args.use_popdist or (args.use_popdist and args.popdist_rank == 0):
            wandb.init(
                project=args.wandb_project_name,
                name=args.wandb_run_name,
                settings=wandb.Settings(console="wrap"),
                config=vars(args),
            )
            if args.half:
                wandb.config.update({"precision": "16.16"})
            else:
                wandb.config.update({"precision": "32.32"})
    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
    main(args)
