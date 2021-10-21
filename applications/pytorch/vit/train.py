# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

import datetime
import os
import time
import warnings
from math import ceil
import poptorch
import torch

from args import parse_args
from checkpoint import restore_checkpoint, save_checkpoint
from datasets import dataset
from ipu_options import get_options
from metrics import accuracy
from log import Logger
from model import PipelinedViTForImageClassification
from optimization import get_lr_scheduler, get_optimizer
import transformers


if __name__ == "__main__":

    # Ignore known warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    # Build config from args
    config = transformers.ViTConfig(**vars(parse_args()))

    # Check output dir
    abs_pathd = os.path.abspath(config.checkpoint_dir)
    os.makedirs(abs_pathd, exist_ok=True)

    log = Logger(abs_pathd+"/"+datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')+'.log',
                 level='INFO')

    # W&B
    if config.wandb:
        import wandb
        proj_name = config.wandb_project_name
        wandb.init(project=proj_name, settings=wandb.Settings(console='off'))
        wandb.config.update(vars(config))

    # Execution parameters
    opts = get_options(config)

    # Dataloader
    train_loader = dataset.get_data(config, opts, train=True, async_dataloader=True)
    test_loader = dataset.get_data(config, opts, train=False, async_dataloader=True)

    steps_per_epoch = len(train_loader)
    if steps_per_epoch < 1:
        raise RuntimeError("Not enough data in input_files for current configuration")

    # IPU Model and Optimizer
    model = PipelinedViTForImageClassification.from_pretrained(config.checkpoint_file, config=config).train()
    if config.precision.startswith("16."):
        model.half()
    optimizer = get_optimizer(config, model)
    scheduler = get_lr_scheduler(optimizer, config.lr_schedule,
                                 config.warmup_steps, config.training_steps)

    epochs_finished = 0
    if config.restore:
        # Retrieve relevant checkpoint
        checkpoint = restore_checkpoint(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        if config.restore_epochs_and_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epochs_finished = checkpoint["epoch"]
            scheduler = get_lr_scheduler(optimizer, config.lr_schedule, config.warmup_steps,
                                         config.training_steps, epochs_finished*steps_per_epoch)
            optimizer._step_count = epochs_finished * steps_per_epoch
            checkpoint_metrics = checkpoint["metrics"]
    else:
        log.logger.info("Training from scratch")

    train_model = poptorch.trainingModel(model, opts, optimizer=optimizer)

    # Compile model
    log.logger.info("---------- Compilation Started ---------")
    start_compile = time.perf_counter()
    datum = dataset.get_random_datum(config)
    train_model.compile(*datum)
    duration_compilation = time.perf_counter() - start_compile
    log.logger.info(f"Compiled model in {duration_compilation} secs")
    log.logger.info("---------------------------------------")

    # Training loop
    log.logger.info("---------- Training Started -----------")

    epochs = ceil(config.training_steps / steps_per_epoch)
    training_steps = config.training_steps
    start_train = time.perf_counter()
    start_step = time.perf_counter()
    for epoch in range(epochs_finished, epochs):
        for step, (input_data, labels) in enumerate(train_loader):
            current_step = step + epoch * steps_per_epoch
            losses, logits = train_model(input_data, labels)
            scheduler.step()
            train_model.setOptimizer(optimizer)
            step_length = time.perf_counter() - start_step
            mean_loss = losses.mean().item()
            preds = torch.argmax(logits, dim=-1)
            acc = accuracy(preds, labels)
            step_throughput = config.samples_per_step / step_length
            msg = ("Epoch: {:.2f}/{} "
                   "Step: {}/{} "
                   "Lr: {:.6f} "
                   "Loss: {:.3f} "
                   "Acc: {:.3f} "
                   "Throughput: {:.2f} samples/sec"
                   ).format(epoch, epochs,
                            current_step, training_steps,
                            scheduler.get_last_lr()[0],
                            mean_loss,
                            acc,
                            step_throughput)
            log.logger.info(msg)
            if config.wandb:
                wandb.log({"LR": scheduler.get_last_lr()[0],
                           "Throughput": step_throughput,
                           "Loss": mean_loss,
                           "Accuracy": acc})

            start_step = time.perf_counter()
            if current_step + 1 == training_steps:
                break  # Training finished mid-epoch
            save_every = current_step % config.checkpoint_save_steps == 0
            not_finished = (current_step + 1 != training_steps)
            if save_every and not_finished:
                filename = save_checkpoint(config, model, optimizer, epoch + 1,
                                           metrics={"Loss": mean_loss})
                log.logger.info("Save checkpoint path: {}".format(filename))

    stop_train = time.perf_counter()
    # Checkpoint at end of run
    save_path = save_checkpoint(config, model, optimizer, epoch + 1,
                                metrics={"Loss": mean_loss})
    log.logger.info("Save checkpoint path: {}".format(save_path))
    log.logger.info("---------------------------------------")

    log.logger.info("---------- Training Metrics -----------")
    log.logger.info(f"global_batch_size: {config.global_batch_size}")
    log.logger.info(f"batches_per_step: {config.batches_per_step}")
    log.logger.info(f"training_steps: {training_steps}")
    duration_run = stop_train - start_train
    num_samples = config.samples_per_step * training_steps
    log.logger.info(f"Training time: {duration_run:.3f} secs")
    log.logger.info("Throughput: {:5f} samples/sec.".format(num_samples / duration_run))
    log.logger.info("---------------------------------------")
