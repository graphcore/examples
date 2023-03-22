# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import datetime
import os
import time
import warnings

import poptorch
import torch
import wandb

from args import parse_args
from checkpoint import load_checkpoint_passing_constraints, prepare_checkpoint_metrics, save_model
from datasets import build_loaders
from ipu_options import get_options
from log import Logger
from model import CLIP
from optimization import get_lr_scheduler, get_optimizer


if __name__ == "__main__":

    # Ignore known warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    # Build config from args
    config = parse_args()

    # Check output dir
    abs_pathd = os.path.abspath(config.checkpoint_dir)
    os.makedirs(abs_pathd, exist_ok=True)

    log = Logger("./output/" + datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S") + ".log", level="info")

    # W&B
    if config.wandb:
        wandb.init(project="torch-clip", name=config.wandb_run_name, settings=wandb.Settings(console="off"))
        wandb.config.update(vars(config))

    # Execution parameters
    opts = get_options(config)

    # Dataloader
    train_loader = build_loaders(config=config, opts=opts, async_dataloader=True)

    steps_per_epoch = len(train_loader)
    if steps_per_epoch < 1:
        raise RuntimeError("Not enough data in input_files for current configuration")

    # IPU Model and Optimizer
    model = CLIP(config).parallelize(log).half().train()
    optimizer = get_optimizer(config, model)

    start_epoch = 0
    epochs = config.epochs
    factor = config.gradient_accumulation * config.device_iterations
    training_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs

    scheduler = get_lr_scheduler(optimizer, config.lr_schedule, warmup_steps, training_steps)

    # Restore model from checkpoint
    if config.restore:
        # Retrieve relevant checkpoint
        checkpoint = load_checkpoint_passing_constraints(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        checkpoint_metrics = checkpoint["metrics"]
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        log.logger.info("Training from sratch")

    # Wrap the training model
    train_model = poptorch.trainingModel(model, opts, optimizer=optimizer)

    # Compile model
    log.logger.info("---------- Compilation Started ---------")
    start_compile = time.perf_counter()
    datum = next(iter(train_loader))
    train_model.compile(*datum)
    duration_compilation = time.perf_counter() - start_compile
    log.logger.info(f"Compiled model in {duration_compilation} secs")
    log.logger.info("---------------------------------------")

    # Track approx. IPU compute time
    total_compute_time = 0

    # Training loop
    log.logger.info("---------- Training Started -----------")
    for epoch in range(start_epoch, epochs):
        start_step = time.perf_counter()
        for step, (image, input_ids) in enumerate(train_loader):
            current_step = step + epoch * steps_per_epoch
            losses = train_model(image, input_ids)
            scheduler.step()
            train_model.setOptimizer(optimizer)
            step_length = time.perf_counter() - start_step
            step_throughput = config.samples_per_step / step_length

            if step > 0 or epoch > 0:
                total_compute_time += step_length

            log.logger.info(
                "Epoch: {:.2f}/{} Step: {}/{} Lr: {:.6f} loss: {:.3f} throughput: {:.2f} samples/sec".format(
                    epoch,
                    epochs,
                    current_step,
                    training_steps,
                    scheduler.get_last_lr()[0],
                    losses.mean(),
                    step_throughput,
                )
            )
            if config.wandb:
                wandb.log({"LR": scheduler.get_last_lr()[0], "Throughput": step_throughput, "Loss": losses})

            if not config.ipu_generate_data and not config.host_generate_data:
                if current_step % config.checkpoint_save_steps == 0 and (current_step + 1 != training_steps):
                    log.logger.info(
                        "Save checkpoint path: {}".format(
                            save_model(
                                config,
                                model,
                                optimizer,
                                epoch + 1,
                                metrics=prepare_checkpoint_metrics(losses, factor),
                                scheduler=scheduler,
                            )
                        )
                    )
            start_step = time.perf_counter()

    # Checkpoint at end of run
    if not config.ipu_generate_data and not config.host_generate_data:
        save_path = save_model(
            config, model, optimizer, epoch + 1, metrics=prepare_checkpoint_metrics(losses, factor), scheduler=scheduler
        )
        log.logger.info("Save checkpoint path: {}".format(save_path))
    log.logger.info("---------------------------------------")
    log.logger.info("---------- Training Metrics -----------")
    log.logger.info(f"global_batch_size: {config.global_batch_size}")
    log.logger.info(f"device_iterations: {config.device_iterations}")
    log.logger.info(f"training_steps: {training_steps}")
    num_samples = config.samples_per_step * (training_steps - 1)
    log.logger.info(f"Training time: {total_compute_time:.3f} secs")
    log.logger.info("throughput: {:5f} samples/sec.".format(num_samples / total_compute_time))
    log.logger.info("---------------------------------------")
