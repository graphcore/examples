# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import time
import torch
import poptorch
import logging
import popdist
import horovod.torch as hvd
from poptorch.optim import SGD, RMSprop, AdamW
import pytorch_lightning as pl
from pytorch_lightning.strategies import IPUStrategy
import sys
import os
import wandb

sys.path.append("../pytorch")
sys.path.append("../pytorch/train")
sys.path.append("../pytorch/utils")

from train_utils import parse_arguments
from validate import create_validation_opts
import models
import utils
import datasets
import datetime

from lr_schedule import WarmUpLRDecorator, PeriodicLRDecorator
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR


class LightningResnet(pl.LightningModule):
    def __init__(self, model, run_opts, label_smoothing=0.0, optimizer=None, lr_scheduler=None):
        super().__init__()
        self.model = model
        self.label_smoothing = 1.0 - label_smoothing
        self.loss = torch.nn.NLLLoss(reduction="mean")
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch_time = time.perf_counter()
        self.step_time = time.perf_counter()
        self.opts = run_opts

    def training_step(self, batch, _):
        input, label = batch
        # Calculate loss in full precision
        output = self.model(input).float()
        log_preds = torch.nn.functional.log_softmax(output, dim=1)
        loss_items = {}
        loss_items["classification_loss"] = self.label_smoothing * self.loss(log_preds, label)
        if self.label_smoothing > 0.0:
            # cross entropy between uniform distribution and output distribution
            loss_items["smoothing_loss"] = -torch.mean(log_preds) * self.label_smoothing
        else:
            loss_items["smoothing_loss"] = torch.zeros(1)
        final_loss = loss_items["smoothing_loss"] + loss_items["classification_loss"]
        return poptorch.identity_loss(final_loss, reduction="mean")

    def training_epoch_end(self, out):
        newtime = time.perf_counter()
        elapsed_time = newtime - self.epoch_time
        self.epoch_time = newtime
        # Need to ensure that if we are using popdist we only log one instance
        if hasattr(self.opts, "use_popdist") and self.opts.use_popdist:
            if self.opts.popdist_rank == 0:
                utils.Logger.log_train_results({"Epoch_elapsed_time": elapsed_time})
                print(f"Epoch_elapsed_time: {elapsed_time}")
        else:
            utils.Logger.log_train_results({"Epoch elapsed time": elapsed_time})
            print(f"Epoch_elapsed_time: {elapsed_time}")

        return

    def training_step_end(self, out):
        newtime = time.perf_counter()
        elapsed_time = newtime - self.step_time
        self.step_time = newtime
        if hasattr(self.opts, "use_popdist") and self.opts.use_popdist:
            if self.opts.popdist_rank == 0:
                utils.Logger.log_train_results({"Step elapsed time": elapsed_time})
        else:
            utils.Logger.log_train_results({"Step elapsed time": elapsed_time})
        return out

    def validation_step(self, batch, _):
        input, labels = batch
        output = self.model(input).float()
        return utils.accuracy(output, labels)

    # Print the validation accuracy only on an epoch level.
    def validation_epoch_end(self, outputs) -> None:
        if hasattr(self.opts, "use_popdist") and self.opts.use_popdist:
            if self.opts.popdist_rank == 0:
                utils.Logger.log_validate_results({"Validation_accuracy": torch.stack(outputs).mean()})
                print(f"Validation_accuracy: {torch.stack(outputs).mean()}")
        else:
            utils.Logger.log_validate_results({"Validation_accuracy": torch.stack(outputs).mean()})
            print(f"Validation_accuracy: {torch.stack(outputs).mean()}")
        return

    def forward(self, input, hidden):
        return self.network(input, hidden)

    def configure_optimizers(self):
        return [self.optimizer], [self.lr_scheduler]


class PopDataLoader(pl.LightningDataModule):
    def __init__(self, run_opts, model_opts, validation_opts):
        super().__init__()
        self.run_opts = run_opts
        self.model_opts = model_opts
        self.validation_opts = validation_opts

    def train_dataloader(self):
        train_data = datasets.get_data(self.run_opts, self.model_opts, train=True, async_dataloader=True)
        return train_data

    def val_dataloader(self):
        test_data = datasets.get_data(
            self.run_opts, self.validation_opts, train=False, async_dataloader=True
        )  # , return_remaining=False)
        return test_data


def create_model_opts(opts):
    if opts.use_popdist:
        model_opts = popdist.poptorch.Options(ipus_per_replica=len(opts.pipeline_splits) + 1)
    else:
        model_opts = poptorch.Options()
        model_opts.replicationFactor(opts.replicas)
    model_opts.deviceIterations(opts.device_iterations)
    # Set mean reduction
    model_opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)

    model_opts.Training.gradientAccumulation(opts.gradient_accumulation)
    if opts.seed is not None:
        model_opts.randomSeed(opts.seed)
    return model_opts


def get_optimizer(opts, model):
    regularized_params = []
    non_regularized_params = []
    for param in model.parameters():
        if param.requires_grad:
            if len(param.shape) == 1:
                non_regularized_params.append(param)
            else:
                regularized_params.append(param)

    params = [
        {"params": regularized_params, "weight_decay": opts.weight_decay},
        {"params": non_regularized_params, "weight_decay": 0},
    ]

    optimizer = None
    if opts.optimizer == "sgd":
        optimizer = SGD(params, lr=opts.lr, momentum=opts.momentum, loss_scaling=opts.initial_loss_scaling)
    elif opts.optimizer == "sgd_combined":
        optimizer = SGD(
            params,
            lr=opts.lr,
            momentum=opts.momentum,
            loss_scaling=opts.initial_loss_scaling,
            velocity_scaling=opts.initial_loss_scaling,
            use_combined_accum=True,
        )
    elif opts.optimizer == "adamw":
        optimizer = AdamW(params, lr=opts.lr, loss_scaling=opts.initial_loss_scaling, eps=opts.optimizer_eps)
    elif opts.optimizer == "rmsprop":
        optimizer = RMSprop(
            params,
            lr=opts.lr,
            alpha=opts.rmsprop_decay,
            momentum=opts.momentum,
            loss_scaling=opts.initial_loss_scaling,
            eps=opts.optimizer_eps,
        )
    elif opts.optimizer == "rmsprop_tf":
        optimizer = RMSprop(
            params,
            lr=opts.lr,
            alpha=opts.rmsprop_decay,
            momentum=opts.momentum,
            loss_scaling=opts.initial_loss_scaling,
            eps=opts.optimizer_eps,
            use_tf_variant=True,
        )

    # Make optimizers distributed
    if opts.use_popdist:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    return optimizer


def get_lr_scheduler(opts, optimizer, step_per_epoch, start_epoch=0):
    scheduler_freq = opts.lr_scheduler_freq if opts.lr_scheduler_freq > 0.0 else step_per_epoch
    scheduler_last_epoch = (scheduler_freq * start_epoch) - 1
    if opts.lr_schedule == "step":
        lr_scheduler = MultiStepLR(
            optimizer=optimizer,
            milestones=[step * scheduler_freq for step in opts.lr_epoch_decay],
            gamma=opts.lr_decay,
            last_epoch=scheduler_last_epoch,
        )
    elif opts.lr_schedule == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer=optimizer, T_max=opts.epoch * scheduler_freq, last_epoch=scheduler_last_epoch
        )
    elif opts.lr_schedule == "exponential":
        lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=opts.lr_decay, last_epoch=scheduler_last_epoch)

    lr_scheduler = PeriodicLRDecorator(optimizer=optimizer, lr_scheduler=lr_scheduler, period=1.0 / scheduler_freq)
    lr_scheduler = WarmUpLRDecorator(optimizer=optimizer, lr_scheduler=lr_scheduler, warmup_epoch=opts.warmup_epoch)
    return lr_scheduler


if __name__ == "__main__":
    run_opts = parse_arguments()

    model_opts = create_model_opts(run_opts)
    logging.info("Loading the data")
    validation_opts = create_validation_opts(run_opts, run_opts.use_popdist)
    len_data = 16
    datamodule = PopDataLoader(run_opts, model_opts, validation_opts)
    len_data = (
        run_opts.micro_batch_size
        * model_opts.device_iterations
        * model_opts.replication_factor
        * model_opts.Training.gradient_accumulation
        * model_opts.Distributed.num_distributed_processes
    )

    logging.info(f"Global batch size, {len_data}")
    logging.info("Initialize the model")
    model = models.get_model(run_opts, datasets.datasets_info[run_opts.data], pretrained=False)
    model.train()

    optimizer = get_optimizer(run_opts, model)

    model_opts = utils.train_settings(run_opts, model_opts)
    lr_scheduler = get_lr_scheduler(run_opts, optimizer, len_data)

    logger = False

    trainer = pl.Trainer(
        accelerator="ipu",
        max_epochs=run_opts.epoch,
        log_every_n_steps=1,
        devices=1,
        accumulate_grad_batches=run_opts.gradient_accumulation,
        strategy=IPUStrategy(inference_opts=validation_opts, training_opts=model_opts, autoreport=False),
        enable_progress_bar=False,
        logger=logger,
    )

    model = LightningResnet(model, run_opts, run_opts.label_smoothing, optimizer, lr_scheduler)

    trainer.fit(model, datamodule)
