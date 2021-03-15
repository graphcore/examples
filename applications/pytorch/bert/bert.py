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
import wandb
import warnings
from tqdm import trange, tqdm
from math import ceil
import logging

import torch
import transformers
from poptorch import trainingModel, DataLoader
from poptorch.enums import DataLoaderMode
from bert_data import get_dataset, get_generated_datum
from bert_model import PipelinedBertWithLoss
from bert_ipu import get_options
from bert_optimization import get_lr_scheduler, get_optimizer
from bert_checkpoint import save_model, maybe_load_checkpoint_passing_constraints, prepare_checkpoint_metrics
from utils import parse_bert_args


if __name__ == "__main__":

    # Ignore known warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    logging.getLogger("poptorch::python").setLevel(logging.ERROR)

    # Build config from args
    config = transformers.BertConfig(**(vars(parse_bert_args())))

    # Retrieve relevant checkpoint
    checkpoint = maybe_load_checkpoint_passing_constraints(config)

    # W&B
    if config.wandb:
        wandb.init(project="torch-bert")
        wandb.config.update(vars(config))

    # Execution parameters
    opts = get_options(config)

    # Dataset selection
    dataset = get_dataset(config)

    # Dataloader
    print("---------- Data Loading Started ---------")
    start_loading = time.perf_counter()
    loader = DataLoader(opts,
                        dataset,
                        batch_size=config.batch_size,
                        num_workers=config.dataloader_workers,
                        mode=DataLoaderMode.Async if config.async_dataloader else DataLoaderMode.Sync)

    steps_per_epoch = len(loader)
    if steps_per_epoch < 1:
        raise RuntimeError("Not enough data in input_files for current configuration, "
                           "try reducing deviceIterations or gradientAccumulation.")
    duration_loader = time.perf_counter() - start_loading
    print(f"Data loaded in {duration_loader} secs")
    print("-----------------------------------------")

    # IPU Model and Optimizer
    model = PipelinedBertWithLoss(config).half().train()
    optimizer = get_optimizer(config, model)
    scheduler = get_lr_scheduler(optimizer, config.lr_schedule,
                                 config.lr_warmup, config.training_steps)

    # Restore model from checkpoint
    epochs_finished = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if config.restore_epochs_and_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.last_epoch = epochs_finished = checkpoint["epoch"]
            checkpoint_metrics = checkpoint["metrics"]
        else:
            # Checkpoint model with epochs and optimizer state reset
            # for further training
            save_model(config, model, optimizer, epochs_finished)
    else:
        # Checkpoint model at start of run
        save_model(config, model, optimizer, epochs_finished)

    poptorch_model = trainingModel(model, opts, optimizer=optimizer)

    # Compile model
    print("---------- Compilation Started ---------")
    start_compile = time.perf_counter()
    datum = get_generated_datum(config)
    poptorch_model.compile(*datum)
    duration_compilation = time.perf_counter() - start_compile
    print(f"Compiled model in {duration_compilation} secs")
    print("---------------------------------------")

    # Training loop
    print("---------- Training Started -----------")

    factor = config.gradient_accumulation * config.batches_per_step
    epochs = ceil(config.training_steps / steps_per_epoch) - epochs_finished
    training_steps = config.training_steps - (steps_per_epoch * epochs_finished)
    start_train = time.perf_counter()
    train_iterator = trange(epochs, desc="Training")
    for epoch in train_iterator:
        epoch_iterator = tqdm(loader, desc="Epoch")
        start_step = time.perf_counter()
        for step, datum in enumerate(epoch_iterator):
            current_step = step + epoch * steps_per_epoch
            outputs = poptorch_model(*datum)
            scheduler.step()
            poptorch_model.setOptimizer(optimizer)
            step_length = time.perf_counter() - start_step
            step_throughput = config.samples_per_step / step_length

            epoch_iterator.set_description(
                f"Epoch: {epoch} / {epochs} - "
                f"LR: {scheduler.get_last_lr()[0]:.2e} - "
                f"Step: {current_step} / {training_steps} - "
                f"Loss: {outputs[0].div(factor).mean().item():3.3f} - "
                f"Loss/MLM: {outputs[1].div(factor).mean().item():3.3f} - "
                f"Loss/NSP: {outputs[2].div(factor).mean().item():3.3f} - "
                f"Acc/MLM: {outputs[3].div(factor).mean().item():3.3f} - "
                f"Acc/NSP: {outputs[4].div(factor).mean().item():3.3f}")

            epoch_iterator.set_postfix_str(f"{step_throughput:.1f} sequences/s")
            if config.wandb:
                wandb.log({"Loss": outputs[0].div(factor).mean().item(),
                           "Loss/MLM": outputs[1].div(factor).mean().item(),
                           "Loss/NSP": outputs[2].div(factor).mean().item(),
                           "Acc/MLM": outputs[3].div(factor).mean().item(),
                           "Acc/NSP": outputs[4].div(factor).mean().item(),
                           "LR": scheduler.get_last_lr()[0],
                           "Epoch": epoch,
                           "Throughput": step_throughput})

            start_step = time.perf_counter()
            if current_step + 1 == training_steps:
                break  # Training finished mid-epoch

        if config.checkpoint_every_epoch and (current_step + 1 != training_steps):
            save_model(config, model, optimizer, epoch + epochs_finished + 1,
                       metrics=prepare_checkpoint_metrics(outputs, factor))

    stop_train = time.perf_counter()
    # Checkpoint at end of run
    save_model(config, model, optimizer, epoch + epochs_finished + 1,
               metrics=prepare_checkpoint_metrics(outputs, factor))
    print("---------------------------------------")

    print("---------- Training Metrics -----------")
    print(f"global_batch_size: {config.global_batch_size}")
    print(f"batches_per_step: {config.batches_per_step}")
    print(f"training_steps: {training_steps}")
    duration_run = stop_train - start_train
    num_samples = config.samples_per_step * training_steps
    print(f"Training time: {duration_run:.3f} secs")
    print("---------------------------------------")
