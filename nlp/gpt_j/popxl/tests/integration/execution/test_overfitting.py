#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import time
import logging

import wandb
import numpy as np
from functools import partial
from transformers import AutoTokenizer
from datasets import load_dataset

import popxl
from popxl.utils import to_numpy

from popxl_addons.utils import timer

from data.hf_data_utils import group_texts
from data.mnli_data import prepare_train_dataset, concat_and_transpose
from popxl_addons import TaskSession
from config import GPTJConfig, CONFIG_DIR
from utils.setup import wandb_init, gptj_fine_tuning_setup
from modelling.hf_mapping import hf_mapping_lm_tp

from finetuning import finetuning
from modelling.embedding import GPTJEmbeddingsTP
from modelling.gptj_lm import GPTJLMHeadLossAndGradTP
from data.data_utils import WorkerInit, DistributedSampler, StatefulDataLoader
from utils.utils import tensor_parallel_input, warmup_schedule


def generate_synthetic_data(config: GPTJConfig):
    words = np.random.randint(0, config.model.embedding.vocab_size, (config.model.sequence_length + 1,))
    input_ids = words[0:-1]
    labels = words[1:]
    data = {"input_ids": [input_ids], "labels": [labels]}
    return data


def overfit(config: GPTJConfig, session: TaskSession):
    replicas = session.ir.instance_replication_factor
    n_shards = config.execution.tensor_parallel
    samples_per_step = config.execution.device_iterations * config.training.global_batch_size

    with timer("Data preparation"):
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        tokenizer.add_special_tokens({"pad_token": "<|extratoken_1|>"})  # index 50257
        dataset = prepare_train_dataset(config)
        sampler = DistributedSampler(dataset)
        train_dl = StatefulDataLoader(
            dataset,
            batch_size=samples_per_step,
            drop_last=True,
            num_workers=1,  # TODO 64
            worker_init_fn=WorkerInit(config.model.seed),
            persistent_workers=True,
            collate_fn=concat_and_transpose,
            sampler=sampler,
        )

    lr_schedule = warmup_schedule(
        config.training.steps,
        1e-7,
        config.training.optimizer.learning_rate.maximum,
        config.training.optimizer.learning_rate.warmup_proportion,
    )
    step = 0
    data = next(iter(train_dl))
    for k in data.keys():
        data[k] = np.repeat(data[k], samples_per_step, 0)

    # Attach to device
    with session:
        while True:
            # Training loop
            for data in train_dl:
                start = time.perf_counter()

                data_map = {}
                words = to_numpy(data["input_ids"], session.inputs.words.dtype, copy=False).reshape(
                    -1, *session.inputs.words.shape
                )
                labels = to_numpy(data["labels"], session.inputs.labels.dtype, copy=False).reshape(
                    -1, *session.inputs.labels.shape
                )
                lr = (
                    np.full((session.ir.num_host_transfers, replicas, 1), lr_schedule[step]).astype("float32").squeeze()
                )
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
                total_steps = config.execution.device_iterations * step
                result_str = (
                    f"Step: {total_steps}/{config.training.steps} "
                    f"Loss: {loss:5.3f} "
                    f"Duration: {duration:6.4f} s "
                    f"Throughput: {throughput:6.1f} samples/s "
                )
                logging.info(result_str)
                wandb.log({"Loss": loss, "LR": lr_schedule[step], "Throughput": throughput}, step=total_steps)
            step += 1

            if total_steps >= config.training.steps:
                return


def main():
    # Configuration
    config, args, pretrained = gptj_fine_tuning_setup(CONFIG_DIR / "finetuning.yml", "release", "gptj_6B_1024_pod64")

    config.training.steps = 100
    config.training.optimizer.learning_rate.maximum = 0.0001
    config.training.optimizer.learning_rate.warmup_proportion = 0.2

    # Create the training session
    if config.checkpoint.load or pretrained:
        session = finetuning(config)
    else:
        # initialise weights from scratch
        session = finetuning(config, no_init=False)

    # Load checkpoint/ HF weights
    if config.checkpoint.load is not None:
        session.load_checkpoint(config.checkpoint.load)
    elif pretrained:
        with timer("Loading HF pretrained model to IPU"):
            session.write_variables_data(hf_mapping_lm_tp(config, session, pretrained))
    else:
        logging.info("Weights initialised from scratch.")

    # Overfit
    overfit(config, session)


if __name__ == "__main__":
    main()
