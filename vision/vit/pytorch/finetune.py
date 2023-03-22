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

import time
import warnings
from math import ceil
from pathlib import Path
import poptorch
import torch
import transformers
import wandb

from args import parse_args
from checkpoint import save_checkpoint
from dataset import get_data
from ipu_options import get_options
from metrics import accuracy
from log import logger
from models import PipelinedViTForImageClassification
import mpi_utils
from optimization import get_lr_scheduler, get_optimizer

if __name__ == "__main__":

    # Ignore known warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    # Build config from args
    config = transformers.ViTConfig(**vars(parse_args()))

    # W&B
    if config.wandb:
        if not config.use_popdist or (config.use_popdist and config.popdist_rank == 0):
            wandb.init(
                project=config.wandb_project_name, name=config.wandb_run_name, settings=wandb.Settings(console="wrap")
            )
            wandb.config.update(vars(config))

    # Execution parameters
    opts = get_options(config)

    # Dataloader
    train_loader = get_data(config, opts, train=True, async_dataloader=True)

    steps_per_epoch = len(train_loader)
    if steps_per_epoch < 1:
        raise RuntimeError("Not enough data in input_files for current configuration")

    # IPU Model and Optimizer
    model = (
        PipelinedViTForImageClassification.from_pretrained(config.pretrained_checkpoint, config=config)
        .parallelize()
        .train()
    )
    model.print_device_allocation()
    if config.precision.startswith("16."):
        model.half()
    optimizer = get_optimizer(config, model)
    scheduler = get_lr_scheduler(optimizer, config.lr_schedule, config.warmup_steps, config.training_steps)

    # Restore model from checkpoint
    steps_finished = 0
    if config.pretrained_checkpoint:
        # Load from checkpoint
        model = (
            PipelinedViTForImageClassification.from_pretrained(config.pretrained_checkpoint, config=config)
            .parallelize()
            .half()
            .train()
        )
        optimizer = get_optimizer(config, model)
        scheduler = get_lr_scheduler(optimizer, config.lr_schedule, config.warmup_steps, config.training_steps)

        if config.resume_training_from_checkpoint:
            training_state = torch.load(Path(config.pretrained_checkpoint) / "training_state.pt")
            scheduler.last_epoch = steps_finished = training_state["step"]
            checkpoint_metrics = training_state["metrics"]
    else:
        # Train model from scratch
        logger.info("Training from scratch")
        model = PipelinedViTForImageClassification(config).parallelize().half().train()
        optimizer = get_optimizer(config, model)
        scheduler = get_lr_scheduler(optimizer, config.lr_schedule, config.lr_warmup, config.training_steps)

    train_model = poptorch.trainingModel(model, opts, optimizer=optimizer)

    # Compile model
    logger.info("---------- Compilation Started ---------")
    start_compile = time.perf_counter()
    datum = next(iter(train_loader))
    train_model.compile(*datum)
    duration_compilation = time.perf_counter() - start_compile
    logger.info(f"Compiled model in {duration_compilation} secs")
    logger.info("---------------------------------------")

    # Training loop
    logger.info("---------- Training Started -----------")

    epochs_finished = steps_finished // steps_per_epoch
    epochs = ceil(config.training_steps / steps_per_epoch)
    training_steps = config.training_steps

    start_train = time.perf_counter()
    for epoch in range(epochs_finished, epochs):
        train_iter = iter(train_loader)
        for step in range(len(train_loader)):
            start_step = time.perf_counter()
            input_data, labels = next(train_iter)
            current_step = step + epoch * steps_per_epoch
            data_duration = time.perf_counter() - start_step
            losses, logits = train_model(input_data, labels)
            scheduler.step()
            train_model.setOptimizer(optimizer)

            step_duration = time.perf_counter() - start_step

            mean_loss = losses.mean().item()
            preds = torch.argmax(logits, dim=-1)
            acc = accuracy(preds, labels)
            step_throughput = len(input_data) / step_duration
            data_consumption_ratio = data_duration / step_duration

            if config.use_popdist:
                step_throughput = mpi_utils.mpi_reduce(step_throughput, average=False)
                step_duration = mpi_utils.mpi_reduce(step_duration, average=True)
                data_consumption_ratio = mpi_utils.mpi_reduce(data_consumption_ratio, average=True)

            if not config.use_popdist or (config.use_popdist and config.popdist_rank == 0):
                msg = (
                    "Epoch: {:.2f}/{} "
                    "Step: {}/{} "
                    "Lr: {:.6f} "
                    "loss: {:.3f} "
                    "accuracy: {:.3f} "
                    "throughput: {:.2f} samples/sec "
                    "Mean step duration: {:.2f} second "
                    "Mean data consumption ratio: {:.2f}"
                ).format(
                    epoch,
                    epochs,
                    current_step,
                    training_steps,
                    scheduler.get_last_lr()[0],
                    mean_loss,
                    acc,
                    step_throughput,
                    step_duration,
                    data_consumption_ratio,
                )
                logger.info(msg)
                if config.wandb:
                    wandb.log(
                        {
                            "LR": scheduler.get_last_lr()[0],
                            "Throughput": step_throughput,
                            "Loss": mean_loss,
                            "Accuracy": acc,
                        }
                    )

            if current_step + 1 == training_steps:
                break  # Training finished mid-epoch
            save_every = current_step % config.checkpoint_save_steps == 0
            finished = current_step + 1 == training_steps
            if config.checkpoint_output_dir and (save_every or finished):
                model.deparallelize()
                save_checkpoint(config, model, optimizer, current_step, metrics={"Loss": mean_loss})
                model.parallelize()

    stop_train = time.perf_counter()

    if config.checkpoint_output_dir:
        # Checkpoint at end of run
        model.deparallelize()
        save_checkpoint(config, model, optimizer, training_steps)
    logger.info("---------------------------------------")

    logger.info("---------- Training Metrics -----------")
    logger.info(f"global_batch_size: {config.global_batch_size}")
    logger.info(f"device_iterations: {config.device_iterations}")
    logger.info(f"training_steps: {training_steps}")
    duration_run = stop_train - start_train
    num_samples = config.samples_per_step * (training_steps - steps_finished)
    logger.info(f"Training time: {duration_run:.3f} secs")
    logger.info("---------------------------------------")
