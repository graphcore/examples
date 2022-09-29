#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import time
import logging
from functools import partial

from transformers import AutoTokenizer
import wandb
from torch.utils.data import DataLoader
import numpy as np


import popxl
from popxl.utils import to_numpy
from popxl_addons import TaskSession
from popxl_addons.utils import timer

from config import GPTJConfig, CONFIG_DIR
from utils.setup import gptj_fine_tuning_setup
from finetuning_mnli import finetuning_mnli
from modelling.embedding import GPTJEmbeddingsTP
from modelling.hf_mapping import hf_mapping_lm_tp
from modelling.gptj_lm import GPTJLMHeadLossAndGradTP
from data.data_utils import WorkerInit
from datasets import load_dataset
from utils.utils import tensor_parallel_input, warmup_schedule, suffix_path
from data.mnli_data import form_text, tokenizes_text, concat_and_transpose
from data.hf_data_utils import group_texts


def training(config: GPTJConfig, session: TaskSession, pretrained):
    samples_per_step = config.execution.device_iterations * \
        config.training.global_batch_size
    n_shards = config.execution.tensor_parallel
    replicas = session.ir.replication_factor

    # Load checkpoint or pretrained
    if config.checkpoint.load is not None:
        with timer('Loading pretrained checkpoint from file to IPU'):
            session.load_checkpoint(config.checkpoint.load)
    elif pretrained:
        with timer('Loading HF pretrained model to IPU'):
            session.write_variables_data(
                hf_mapping_lm_tp(config, session, pretrained))
    else:
        logging.info(f"Not loading a pretrained model.")

    with timer('Data preperation'):
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
        tokenizer.add_special_tokens(
            {'pad_token': '<|extratoken_1|>'})  # index 50257
        dataset = load_dataset('glue', 'mnli', split='train')
        dataset = dataset.map(form_text,
                              remove_columns=['hypothesis',
                                              'premise', 'label', 'idx'],
                              load_from_cache_file=True, desc='Generating text prompt')
        dataset = dataset.map(tokenizes_text(tokenizer),
                              batched=True,
                              batch_size=1000,
                              num_proc=1,
                              remove_columns=dataset.column_names,
                              load_from_cache_file=True, desc='Tokenizing text')
        dataset = dataset.map(group_texts(config),
                              batched=True,
                              batch_size=1000,
                              num_proc=1,
                              load_from_cache_file=True, desc='Packing sequences')
        train_dl = DataLoader(dataset,
                              batch_size=samples_per_step,
                              drop_last=True,
                              num_workers=1,  # TODO 64
                              worker_init_fn=WorkerInit(config.model.seed),
                              persistent_workers=True,
                              collate_fn=concat_and_transpose)

    lr_schedule = warmup_schedule(
        config.training.steps, 1e-7,
        config.training.optimizer.learning_rate.maximum,
        config.training.optimizer.learning_rate.warmup_proportion)

    step = 0
    total_steps = 0
    prev_total_steps = 0

    # Attach to device
    with session:
        while True:
            # Training loop
            for data in train_dl:
                start = time.perf_counter()
                saved_checkpoint = False

                data_map = {}
                words = to_numpy(
                    data['input_ids'],
                    session.inputs.words.dtype,
                    copy=False).reshape(
                        -1, *session.inputs.words.shape)
                labels = to_numpy(data['labels'],
                                  session.inputs.labels.dtype, copy=False).reshape(
                                      -1, *session.inputs.labels.shape)
                seeds = popxl.create_seeds(config.model.seed, step,
                                           samples_per_step)
                lr = np.full((session.ir.num_host_transfers,
                              config.execution.data_parallel * n_shards, 1),
                             lr_schedule[step]).astype('float32').squeeze()

                data_map[session.inputs.words] = tensor_parallel_input(
                    words, n_shards, replicas,
                    partial(GPTJEmbeddingsTP.offset_input,
                            config=config))
                data_map[session.inputs.labels] = tensor_parallel_input(
                    labels, n_shards, replicas,
                    partial(GPTJLMHeadLossAndGradTP.offset_input,
                            config=config))
                # Seed (different per DP and identical per TP replica)
                data_map[session.inputs.seed] = tensor_parallel_input(
                    seeds, n_shards, replicas)
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

                # Periodically save checkpoint
                if config.checkpoint.steps > 0:
                    checkpoint_step = total_steps // config.checkpoint.steps
                    prev_checkpoint_step = prev_total_steps // config.checkpoint.steps
                    if checkpoint_step - prev_checkpoint_step >= 1 and total_steps >= config.checkpoint.steps:
                        saved_checkpoint = True
                        path = suffix_path(
                            config.checkpoint.save, f'_train_step_{total_steps}')
                        with timer('Saving training checkpoint'):
                            session.save_checkpoint(path)

                if total_steps >= config.training.steps:
                    # Save last checkpoint
                    if config.checkpoint.save is not None and not saved_checkpoint:
                        path = suffix_path(
                            config.checkpoint.save, f'_train_step_{total_steps}')
                        with timer('Saving last checkpoint'):
                            session.save_checkpoint(path)

                    return

                step += 1


def main():
    # Configuration
    config, args, pretrained = gptj_fine_tuning_setup(
        CONFIG_DIR / "finetuning_mnli.yml", "release", "tiny")

    # Create the training session
    train_session = finetuning_mnli(config)

    # Train
    training(config, train_session, pretrained)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)  # Log time of exception
        raise
