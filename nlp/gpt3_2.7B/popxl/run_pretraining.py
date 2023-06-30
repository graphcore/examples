#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import time
import logging
from itertools import islice
from more_itertools import ncycles
from typing import Optional
from math import ceil

from torch.utils.data import DataLoader, Dataset
from transformers.models.gpt2 import GPT2LMHeadModel as HFGPT2LMHeadModel
import wandb
import numpy as np

from popxl.utils import to_numpy
from popxl_addons import TaskSession
from popxl_addons.utils import timer

from config import GPTConfig, CONFIG_DIR
from utils.setup import gpt_training_setup
from pretraining import pretraining
from modelling.hf_mapping import hf_mapping_lm_TP
from data.data_utils import WorkerInit
from datasets import load_dataset
from utils.utils import linear_schedule, suffix_path
from popxl_addons.array_munging import tensor_parallel_input
from data.data_utils import load_dataset, WorkerInit, collate_fn


def pretraining(
    config: GPTConfig, session: TaskSession, dataset: Dataset, pretrained: Optional[HFGPT2LMHeadModel] = None
):
    samples_per_step = config.execution.device_iterations * config.training.global_batch_size
    tp = config.execution.tensor_parallel
    local_replicas = session.ir.instance_replication_factor
    local_tp_all = tp if local_replicas >= tp else local_replicas
    seq_len = config.model.sequence_length

    session.add_session_state_info({"total_steps": 0})

    with timer("Data preparation"):
        if config.data.n_samples:
            logging.warning(f"Limiting input data to {config.data.n_samples} samples.")
            dataset = islice(iter(dataset), config.data.n_samples)
            dataset = list(ncycles(dataset, ceil(samples_per_step / config.data.n_samples)))
        train_dl = DataLoader(
            dataset,
            batch_size=samples_per_step,
            drop_last=True,
            num_workers=1,  # If >1 data is non-deterministic
            worker_init_fn=WorkerInit(config.model.seed),
            persistent_workers=True,
            collate_fn=collate_fn,
        )

    # Load checkpoint or pretrained
    if config.checkpoint.load is not None:
        with timer("Loading pretrained checkpoint from file to IPU"):
            session.load_checkpoint(config.checkpoint.load)
    elif pretrained:
        with timer("Loading HF pretrained model to IPU"):
            session.write_variables_data(hf_mapping_lm_TP(config, session, pretrained))
    else:
        logging.info(f"Not loading a pretrained model.")

    if config.training.optimizer.learning_rate.constant:
        lr_schedule = {i: config.training.optimizer.learning_rate.maximum for i in range(config.training.steps + 1)}
    else:
        lr_schedule = linear_schedule(
            config.training.steps,
            1e-7,
            config.training.optimizer.learning_rate.maximum,
            config.training.optimizer.learning_rate.warmup_proportion,
        )

    step = session.session_state["steps"]
    total_steps = session.session_state["total_steps"]
    prev_total_steps = session.session_state["total_steps"]
    loop = True
    printed_data_warning = False

    # Attach to device
    logging.info(f"Opening session.")
    with session:
        logging.info(f"Opened session.")
        while loop:
            # Training loop
            for data in train_dl:
                start = time.perf_counter()

                words = data["input_ids"]
                labels = data["labels"]

                assert words.shape[1] == labels.shape[1]
                data_seq_len = words.shape[1]
                assert data_seq_len >= seq_len, (
                    "Data sequence length is too small for model. "
                    f"Data seq len: {data_seq_len}. Model seq len: {seq_len}"
                )

                if data_seq_len > seq_len:
                    if not printed_data_warning:
                        logging.warning(
                            "Data sequence length is larger than model sequence length. Trimming data to fit. "
                            f"Data seq len: {data_seq_len}. Model seq len: {seq_len}"
                        )
                        printed_data_warning = True
                    words = words[:, :seq_len]
                    labels = labels[:, :seq_len]

                data_map = {}
                words = to_numpy(words, session.inputs.words.dtype, copy=False).reshape(-1, *session.inputs.words.shape)
                labels = to_numpy(labels, session.inputs.labels.dtype, copy=False).reshape(
                    -1, *session.inputs.labels.shape
                )
                lr = (
                    np.full((session.ir.num_host_transfers, local_replicas), lr_schedule[step])
                    .astype("float32")
                    .squeeze()
                )

                data_map[session.inputs.words] = tensor_parallel_input(words, local_tp_all, local_replicas)
                data_map[session.inputs.labels] = tensor_parallel_input(labels, local_tp_all, local_replicas)

                # Add learning rate inputs
                data_map[session.inputs.lr] = lr

                # Run program
                outputs = session.run(data_map)
                losses = outputs[session.outputs["loss"]]
                grad_norm = outputs[session.outputs["grad_norm"]][0]

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
                wandb.log(
                    {
                        "Loss": loss,
                        "LR": lr_schedule[step],
                        "Throughput": throughput,
                        "grad_norm": grad_norm,
                    },
                    step=total_steps,
                )
                session.session_state["total_steps"] = total_steps

                # Periodically save checkpoint
                saved_checkpoint = False
                if config.checkpoint.steps > 0:
                    checkpoint_step = total_steps // config.checkpoint.steps
                    prev_checkpoint_step = prev_total_steps // config.checkpoint.steps
                    if checkpoint_step - prev_checkpoint_step >= 1 and total_steps >= config.checkpoint.steps:
                        saved_checkpoint = True
                        path = suffix_path(config.checkpoint.save, f"_train_step_{total_steps}")
                        with timer("Saving training checkpoint"):
                            session.save_checkpoint(path)

                if total_steps >= config.training.steps:
                    # Save last checkpoint
                    if config.checkpoint.save is not None and not saved_checkpoint:
                        path = suffix_path(config.checkpoint.save, f"_train_step_{total_steps}")
                        with timer("Saving last checkpoint"):
                            session.save_checkpoint(path)

                    loop = False
                    break

                step += 1

        logging.info(f"Closing session.")
    logging.info(f"Closed session.")


def main():
    # Configuration
    config, args, pretrained = gpt_training_setup(CONFIG_DIR / "pretraining.yml", "release", "gpt3_2.7B_pod64")

    # Setup dataset
    if not config.data.input_files:
        raise ValueError(f"Input files for data needs to be set.")
    dataset = load_dataset(config.data.input_files)

    # Create the training session
    train_session = pretraining(config)

    # Train
    pretraining(config, train_session, dataset, pretrained)

    logging.info(f"Finished pre-training.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e, exc_info=False)  # Log time of exception
        raise
