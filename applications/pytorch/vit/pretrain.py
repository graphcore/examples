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
import wandb
import horovod.torch as hvd

from args import parse_args
from checkpoint import restore_checkpoint, save_checkpoint
from dataset import get_dataset, get_dataloader, mixup_data, get_random_datum, DatasetWithStepLabel
from ipu_options import get_options
from log import logger
from metrics import accuracy
from models import PipelinedViTForImageClassificationPretraining
import mpi_utils
from optimization import get_lr_scheduler, get_optimizer


if __name__ == "__main__":

    # Ignore known warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    # Build config from args
    config = parse_args()

    # W&B
    if config.wandb:
        if not config.use_popdist or (config.use_popdist and config.popdist_rank == 0):
            proj_name = config.wandb_project_name
            wandb.init(project=proj_name,
                       settings=wandb.Settings(console='off'))
            wandb.config.update(vars(config))

    # Execution parameters
    opts = get_options(config)

    # Dataloader
    logger.info('Loading data ... ')
    train_dataset = get_dataset(config, opts, train=True)
    train_dataset = DatasetWithStepLabel(
        train_dataset, config.samples_per_step, config.random_seed)

    train_dataloader = get_dataloader(
        config, opts, train_dataset, train=True, async_dataloader=True)

    steps_per_epoch = len(train_dataloader)
    if steps_per_epoch < 1:
        raise RuntimeError(
            "Not enough data in input_files for current configuration")

    if config.epochs > 0:
        logger.info("Reconfiguring training steps according to epochs: %d"
                    % config.epochs)
        config.training_steps = config.epochs * steps_per_epoch

    # IPU Model and Optimizer
    model = PipelinedViTForImageClassificationPretraining(config).train()
    if config.precision[-3:] == ".16":
        logger.info('Setting model to half precision')
        model = model.half()

    optimizer = get_optimizer(config, model)
    scheduler = get_lr_scheduler(optimizer, config.lr_schedule,
                                 config.warmup_steps, config.training_steps)

    epochs_finished = 0
    if config.resume_training_from_checkpoint:
        # Retrieve relevant checkpoint
        model_state_dict, training_state = restore_checkpoint(config)
        model.load_state_dict(model_state_dict)
        logger.info("Weights are restored")
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        epochs_finished = training_state["epoch"]
        scheduler = get_lr_scheduler(optimizer, config.lr_schedule, config.warmup_steps,
                                     config.training_steps, epochs_finished*steps_per_epoch)
        optimizer._step_count = epochs_finished * steps_per_epoch
        checkpoint_metrics = training_state["metrics"]
        logger.info("Epochs and optimizer_state_dict are restored")
    else:
        logger.info("Training from scratch")

    train_model = poptorch.trainingModel(model, opts, optimizer=optimizer)

    # Compile model
    logger.info("---------- Compilation Started ---------")
    start_compile = time.perf_counter()
    datum = get_random_datum(config)
    if config.mixup:
        input_data, labels = datum[0], datum[1]
        input_data, labels_a, labels_b, lam = mixup_data(
            input_data, labels, config.alpha)
        datum = [input_data, labels_a, labels_b, lam]
        if config.byteio:
            input_data = torch.clip(input_data.float()*255, 0, 255).byte()
        datum = [input_data, labels_a, labels_b, lam]
    train_model.compile(*datum)
    duration_compilation = time.perf_counter() - start_compile
    logger.info(f"Compiled model in {duration_compilation} secs")
    logger.info("---------------------------------------")

    # Training loop
    logger.info("---------- Training Started -----------")

    epochs = ceil(config.training_steps / steps_per_epoch)
    training_steps = config.training_steps
    current_step = -1

    start_train = time.perf_counter()
    for epoch in range(epochs_finished, epochs):
        train_iter = iter(train_dataloader)
        for _ in range(len(train_dataloader)):
            start_step = time.perf_counter()
            train_data = next(train_iter)
            current_step += 1
            data_duration = time.perf_counter() - start_step

            # train_data format when mixup is True / False
            # input_data, labels_a, labels_b, lam / input_data, labels
            losses, logits = train_model(*train_data)
            scheduler.step()
            train_model.setOptimizer(optimizer)
            step_duration = time.perf_counter() - start_step

            mean_loss = losses.mean().item()
            preds = torch.argmax(logits, dim=-1)
            acc = accuracy(preds, *train_data[1:])
            num_input_samples = len(train_data[0])
            step_throughput = num_input_samples / step_duration
            data_consumption_ratio = data_duration / step_duration
            if config.use_popdist:
                step_throughput = mpi_utils.mpi_reduce(
                    step_throughput, average=False)
                step_duration = mpi_utils.mpi_reduce(
                    step_duration, average=True)
                data_consumption_ratio = mpi_utils.mpi_reduce(
                    data_consumption_ratio, average=True)

            if not config.use_popdist or (config.use_popdist and config.popdist_rank == 0):
                msg = ("Epoch: {:.2f}/{} "
                       "Step: {}/{} "
                       "Lr: {:.6f} "
                       "Loss: {:.3f} "
                       "Acc: {:.3f} "
                       "Throughput: {:.2f} samples/sec "
                       "Mean step duration: {:.2f} seconds "
                       "Mean data consumption ratio: {:.2f}"
                       ).format(epoch, epochs,
                                current_step,
                                training_steps,
                                scheduler.get_last_lr()[0],
                                mean_loss,
                                acc,
                                step_throughput,
                                step_duration,
                                data_consumption_ratio)
                logger.info(msg)
                if config.wandb:
                    wandb.log({"LR": scheduler.get_last_lr()[0],
                               "Throughput": step_throughput,
                               "Loss": mean_loss,
                               "Accuracy": acc})

            start_step = time.perf_counter()
            if current_step + 1 == training_steps:
                break  # Training finished mid-epoch
            save_every = current_step % config.checkpoint_save_steps == 0
            finished = (current_step + 1 == training_steps)

            if config.checkpoint_output_dir and (save_every or finished):
                save_checkpoint(config, model, optimizer, current_step,
                                metrics={"Loss": mean_loss})

    stop_train = time.perf_counter()
    # Checkpoint at end of run
    save_path = save_checkpoint(config, model, optimizer, current_step,
                                metrics={"Loss": mean_loss})
    logger.info("---------------------------------------")

    logger.info("---------- Training Metrics -----------")
    logger.info(f"global_batch_size: {config.global_batch_size}")
    logger.info(f"batches_per_step: {config.batches_per_step}")
    logger.info(f"training_steps: {training_steps}")
    duration_run = stop_train - start_train
    num_samples = config.samples_per_step * training_steps
    logger.info(f"Training time: {duration_run:.3f} secs")
    logger.info("---------------------------------------")
    time.sleep(2)  # wait for child processes to join
