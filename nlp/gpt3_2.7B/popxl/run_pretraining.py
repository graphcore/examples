#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import time
import logging

import wandb
from torch.utils.data import DataLoader, Dataset
import numpy as np
from functools import partial

import popxl
from popxl.utils import to_numpy
from popxl_addons import TaskSession

from config import GPTConfig, CONFIG_DIR
from utils.setup import wandb_init, gpt_pretraining_setup
from pretraining import pretraining
from modelling.embedding import GPTEmbeddingsTP
from data.data_utils import load_dataset, WorkerInit, collate_fn
from utils.utils import tensor_parallel_input, linear_schedule


def training(config: GPTConfig, session: TaskSession, dataset: Dataset):
    samples_per_step = config.execution.device_iterations * config.training.global_batch_size

    n_shards = config.execution.tensor_parallel
    # Shape should be: [ir.num_host_transfers, ir.replication_factor, *t.shape]
    # (session.ir.num_host_transfers = config.execution.device_iterations * config.gradient_accumulation)
    shape_prefix = (session.ir.num_host_transfers, )

    replicas = session.ir.replication_factor

    train_dl = DataLoader(dataset,
                          batch_size=samples_per_step,
                          drop_last=True,
                          num_workers=64,
                          worker_init_fn=WorkerInit(config.model.seed),
                          persistent_workers=True,
                          collate_fn=collate_fn)

    step = 0

    lr_schedule = linear_schedule(
        config.training.steps, 1e-7,
        config.training.optimizer.learning_rate.maximum,
        config.training.optimizer.learning_rate.warmup_proportion)

    # Attach to device
    with session:
        start = time.perf_counter()
        while True:
            # Training loop
            for data in train_dl:
                data_map = {}
                words = to_numpy(
                    data['input_ids'][:, :config.model.sequence_length],
                    session.inputs.words.dtype).reshape(
                        -1, *session.inputs.words.shape)
                labels = to_numpy(data['labels'],
                                  session.inputs.labels.dtype).reshape(
                                      -1, *session.inputs.labels.shape)
                lr = np.full((session.ir.num_host_transfers,
                              config.execution.data_parallel * n_shards, 1),
                             lr_schedule[step]).astype('float32').squeeze()

                data_map[session.inputs.words] = tensor_parallel_input(
                    words, n_shards, replicas,
                    partial(GPTEmbeddingsTP.offset_input,
                            vocab_size=config.model.embedding.vocab_size,
                            n_shards=n_shards))
                data_map[session.inputs.labels] = tensor_parallel_input(
                    labels, n_shards, replicas,
                    partial(GPTEmbeddingsTP.offset_input,
                            vocab_size=config.model.embedding.vocab_size,
                            n_shards=n_shards))
                # Add learning rate inputs
                data_map[session.inputs.lr] = lr

                # Run program
                outputs = session.run(data_map)
                losses = outputs[session.outputs[0]]

                # Logging
                duration = time.perf_counter() - start
                start = time.perf_counter()

                loss = np.mean(losses.astype(np.float32))
                throughput = samples_per_step / duration
                total_steps = config.execution.device_iterations * step
                result_str = (f"Step: {total_steps}/{config.training.steps} "
                              f"Loss: {loss:5.3f} "
                              f"Duration: {duration:6.4f} s "
                              f"Throughput: {throughput:6.1f} samples/s ")
                logging.info(result_str)
                wandb.log(
                    {
                        "Loss": loss,
                        "LR": lr_schedule[step],
                        "Throughput": throughput
                    },
                    step=total_steps)
                step += 1

                if total_steps >= config.training.steps:
                    return


def main():
    # Configuration
    config, args = gpt_pretraining_setup(CONFIG_DIR / "pretraining.yml",
                                         "release", "tiny")

    # Setup weights and biases
    wandb_init(config, tags=['PE', 'TP'], disable=args.wandb == 'False')

    # Setup dataset
    if not config.data.input_files:
        raise ValueError(f"Input files for data needs to be set.")
    dataset = load_dataset(config.data.input_files)

    # Create the training session
    session = pretraining(config)

    # Load checkpoint
    if config.checkpoint.load is not None:
        session.load_checkpoint(config.checkpoint.load)

    # Train
    training(config, session, dataset)

    # Save checkpoint
    if config.checkpoint.save is not None:
        session.save_checkpoint(config.checkpoint.save)


if __name__ == "__main__":
    main()
