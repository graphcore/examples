#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import time
import logging

import wandb
from torch.utils.data import DataLoader
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


def generate_synthetic_data(config: GPTConfig):
    words = np.random.randint(0, config.model.embedding.vocab_size, (config.model.sequence_length+1,))
    input_ids = words[0:-1]
    labels = words[1:]
    data = {'input_ids': [input_ids], 'labels': [labels]}
    return data


def overfit(config: GPTConfig, session: TaskSession):
    rf = session.ir.instance_replication_factor
    n_shards = config.execution.tensor_parallel
    samples_per_step = config.execution.device_iterations * config.training.global_batch_size

    dataset = load_dataset(config.data.input_files)
    train_dl = DataLoader(dataset,
                          batch_size=1,
                          drop_last=True,
                          num_workers=16,
                          worker_init_fn=WorkerInit(config.model.seed),
                          persistent_workers=True,
                          collate_fn=collate_fn)

    lr_schedule = linear_schedule(config.training.steps, 1e-7, config.training.optimizer.learning_rate.maximum,
                                  config.training.optimizer.learning_rate.warmup_proportion)
    step = 0
    data = next(iter(train_dl))
    for k in data.keys():
        data[k] = np.repeat(data[k], samples_per_step, 0)
    # Attach to device
    with session:
        start = time.perf_counter()
        while True:
            # Training loop
            data_map = {}
            # Words, tokens_types, mask, labels
            words = to_numpy(data['input_ids'][:, :config.model.sequence_length], session.inputs.words.dtype).reshape(
                -1, *session.inputs.words.shape)
            labels = to_numpy(data['labels'], session.inputs.labels.dtype).reshape(-1, *session.inputs.labels.shape)
            # Seed (different per DP and identical per TP replica)
            seeds = popxl.create_seeds(config.model.seed, step,
                                       session.ir.num_host_transfers * config.execution.data_parallel).reshape(-1, 2)

            data_map[session.inputs.words] = tensor_parallel_input(
                words, n_shards, rf,
                partial(GPTEmbeddingsTP.offset_input, vocab_size=config.model.embedding.vocab_size,
                        n_shards=n_shards)).squeeze()
            data_map[session.inputs.labels] = tensor_parallel_input(
                labels, n_shards, rf,
                partial(GPTEmbeddingsTP.offset_input, vocab_size=config.model.embedding.vocab_size,
                        n_shards=n_shards)).squeeze()
            data_map[session.inputs.seed] = tensor_parallel_input(seeds, n_shards, rf).squeeze()
            data_map[session.inputs.lr] = np.full(session._full_input_shape(()), lr_schedule[step]).astype(np.float32)

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
                    "Throughput": throughput,
                    "Grad Norm": outputs[session.outputs[1]].flatten()[0]
                },
                step=total_steps)
            step += 1

            if total_steps >= config.training.steps:
                return


def main():
    # Configuration
    config, args = gpt_pretraining_setup(CONFIG_DIR / "pretraining.yml", "release", "gpt3_2.7B_pod64")
    config.training.steps = 100
    config.training.optimizer.learning_rate.maximum = 0.0001
    config.training.optimizer.learning_rate.warmup_proportion = 0.2

    # Setup weights and biases
    wandb_init(config, tags=['overfit'], disable=args.wandb == 'False')

    # Create the training session
    session = pretraining(config)

    # Load checkpoint
    if config.checkpoint.load is not None:
        session.load_checkpoint(config.checkpoint.load)

    # Overfit
    overfit(config, session)


if __name__ == "__main__":
    main()
