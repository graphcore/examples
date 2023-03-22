#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import time
import logging
from functools import partial

import wandb
import numpy as np


import popxl
from popxl.utils import to_numpy
from popxl_addons import TaskSession
from popxl_addons.utils import timer

from config import GPTJConfig, CONFIG_DIR
from utils.setup import gptj_fine_tuning_setup, wandb_init
from finetuning import finetuning
from modelling.embedding import GPTJEmbeddingsTP
from modelling.hf_mapping import hf_mapping_lm_tp
from modelling.gptj_lm import GPTJLMHeadLossAndGradTP
from data.data_utils import WorkerInit
from utils.utils import tensor_parallel_input, warmup_schedule, suffix_path
from data.mnli_data import prepare_train_dataset, concat_and_transpose
from data.data_utils import StatefulDataLoader
from datetime import datetime
import os
import popdist


def training(config: GPTJConfig, session: TaskSession, pretrained, dataset):
    samples_per_step = config.execution.device_iterations * config.training.global_batch_size
    n_shards = config.execution.tensor_parallel
    replicas = session.ir.instance_replication_factor

    # not necessary if the config stays the same between checkpoints
    session.add_session_state_info(
        {
            "lr_schedule": {
                "steps": config.training.steps,
                "min": 1e-7,
                "max": config.training.optimizer.learning_rate.maximum,
                "warmup_proportion": config.training.optimizer.learning_rate.warmup_proportion,
            }
        }
    )

    session.add_session_state_info({"total_steps": 0})

    train_dl = StatefulDataLoader(
        dataset,
        batch_size=samples_per_step,
        drop_last=True,
        num_workers=1,  # TODO 64
        worker_init_fn=WorkerInit(config.model.seed),
        persistent_workers=True,
        collate_fn=concat_and_transpose,
    )

    session.dataloader = train_dl

    # Load checkpoint or pretrained
    if config.checkpoint.load is not None:
        session.load_checkpoint(config.checkpoint.load)
    elif pretrained:
        with timer("Loading HF pretrained model to IPU"):
            session.write_variables_data(hf_mapping_lm_tp(config, session, pretrained))
    else:
        logging.info(f"Not loading a pretrained model.")

    lr_sch = session.session_state["lr_schedule"]

    lr_schedule = warmup_schedule(lr_sch["steps"], lr_sch["min"], lr_sch["max"], lr_sch["warmup_proportion"])

    step = session.session_state["steps"]
    total_steps = session.session_state["total_steps"]
    prev_total_steps = session.session_state["total_steps"]

    # Attach to device
    checkpoint_dir = config.checkpoint.save
    if checkpoint_dir is not None:
        checkpoint_dir = os.path.join(checkpoint_dir, "Run_{}".format(datetime.now().strftime("%d_%m_%Y_%H_%M")))

    while True:
        # Training loop
        # Dataloader automatically count epochs
        for data in train_dl:
            start = time.perf_counter()
            saved_checkpoint = False

            data_map = {}
            words = to_numpy(data["input_ids"], session.inputs.words.dtype, copy=False).reshape(
                -1, *session.inputs.words.shape
            )
            labels = to_numpy(data["labels"], session.inputs.labels.dtype, copy=False).reshape(
                -1, *session.inputs.labels.shape
            )
            lr = np.full((session.ir.num_host_transfers, replicas, 1), lr_schedule[step]).astype("float32").squeeze()

            data_map[session.inputs.words] = tensor_parallel_input(
                words, n_shards, replicas, partial(GPTJEmbeddingsTP.offset_input, config=config)
            )
            data_map[session.inputs.labels] = tensor_parallel_input(
                labels, n_shards, replicas, partial(GPTJLMHeadLossAndGradTP.offset_input, config=config)
            )
            # Add learning rate inputs
            data_map[session.inputs.lr] = lr

            # Run program
            outputs = session.run(data_map)
            losses = outputs[session.outputs[0]]

            # Logging
            duration = time.perf_counter() - start  # Don't include checkpoint saving

            loss = np.mean(losses.astype(np.float32))
            throughput = samples_per_step / duration
            prev_total_steps = total_steps
            total_steps = config.execution.device_iterations * step
            result_str = (
                f"Step: {total_steps}/{config.training.steps} "
                f"Loss: {loss:5.3f} "
                f"Duration: {duration:6.4f} s "
                f"Throughput: {throughput:6.1f} samples/s "
            )
            logging.info(result_str)
            if popdist.getInstanceIndex() == 0 and wandb.run is not None:
                wandb.log({"Loss": loss, "LR": lr_schedule[step], "Throughput": throughput}, step=total_steps)
            session.session_state["total_steps"] = total_steps

            # Periodically save checkpoint
            if config.checkpoint.steps > 0:
                checkpoint_step = total_steps // config.checkpoint.steps
                prev_checkpoint_step = prev_total_steps // config.checkpoint.steps
                if checkpoint_step - prev_checkpoint_step >= 1 and total_steps >= config.checkpoint.steps:
                    saved_checkpoint = True
                    path = os.path.join(checkpoint_dir, f"train_step_{total_steps}")
                    session.save_checkpoint(path)

            if total_steps >= config.training.steps:
                # Save last checkpoint
                if checkpoint_dir is not None and not saved_checkpoint:
                    path = os.path.join(checkpoint_dir, f"train_step_{total_steps}")
                    session.save_checkpoint(path)

                return

            step += 1


def main():
    # Configuration
    config, args, pretrained = gptj_fine_tuning_setup(CONFIG_DIR / "finetuning.yml", "release", "gptj_6B_1024_pod64")

    # Create the training session
    train_session = finetuning(config)

    # Load dataset
    with timer("Data preparation"):
        dataset = prepare_train_dataset(config)

    # Train
    training(config, train_session, pretrained, dataset)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)  # Log time of exception
        raise
