#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import time
import logging

import wandb
import numpy as np
from torch.utils.data import DataLoader, Dataset

import popxl
from popxl_addons import TaskSession

from pretraining import pretraining_phased
from config import BertConfig, CONFIG_DIR
from utils.lr_schedule import linear_schedule
from utils.setup import wandb_init, bert_pretraining_setup
from data.pretraining_data import load_dataset, WorkerInit


def training(config: BertConfig, session: TaskSession, dataset: Dataset):
    samples_per_step = config.execution.device_iterations * config.training.global_batch_size

    train_dl = DataLoader(
        dataset,
        batch_size=samples_per_step,
        drop_last=True,
        num_workers=64,
        worker_init_fn=WorkerInit(config.model.seed),
        persistent_workers=True)

    step = 0

    lr_schedule = linear_schedule(
        config.training.steps,
        1e-7,
        config.training.optimizer.learning_rate.maximum,
        config.training.optimizer.learning_rate.warmup_proportion)

    # Attach to device
    with session:
        start = time.perf_counter()
        while True:
            # Training loop
            for data in train_dl:
                data_map = {}

                # Validate data matches config
                data_seq_len = data['input_ids'].shape[1]
                if data_seq_len != config.model.sequence_length:
                    raise ValueError(f'Sequence length in config ({config.model.sequence_length}) does not match '
                                     f"sequence length in input data ({data_seq_len})')")
                data_mlm_mask_tokens = data['masked_lm_positions'].shape[1]
                if data_mlm_mask_tokens != config.model.mlm.mask_tokens:
                    raise ValueError(f'MLM masked tokens in config ({config.model.mlm.mask_tokens}) does not match '
                                     f"MLM masked tokens in input data ({data_mlm_mask_tokens})')")

                for idx, key in enumerate(['input_ids', 'segment_ids', 'input_mask', 'masked_lm_positions', 'masked_lm_ids', 'next_sentence_labels']):
                    h2d = session.inputs[idx]
                    data_map[h2d] = data[key].numpy()\
                        .astype(h2d.dtype.as_numpy())\
                        .reshape(session.ir.num_host_transfers, config.execution.data_parallel, *h2d.shape)

                seeds = popxl.create_seeds(config.model.seed, step, batches_per_step=session.ir.num_host_transfers, replicas=config.execution.data_parallel)
                data_map[session.inputs[len(data_map)]] = seeds

                # # Add learning rate inputs
                # # TODO: Allow accepting of smaller sized inputs.
                data_map[session.inputs[len(data_map)]] = np.full((session.ir.num_host_transfers, config.execution.data_parallel, 1), lr_schedule[step]).astype(np.float32)

                # Run program
                outputs = session.run(data_map)
                losses = outputs[session.outputs[0]]
                grad_norm = outputs[session.outputs[1]].flatten()[0]

                # Logging
                duration = time.perf_counter() - start
                start = time.perf_counter()

                loss = np.mean(losses.astype(np.float32))
                throughput = samples_per_step/duration
                total_steps = config.execution.device_iterations * step
                result_str = (
                    f"Step: {total_steps}/{config.training.steps} "
                    f"Loss: {loss:5.3f} "
                    f"Duration: {duration:6.4f} s "
                    f"throughput: {throughput:6.1f} samples/sec ")
                logging.info(result_str)
                wandb.log({"Loss": loss,
                           "LR": lr_schedule[step],
                           "Throughput": throughput,
                           "Grad Norm": grad_norm},
                          step=total_steps)
                step += 1

                if total_steps >= config.training.steps:
                    return


def main():
    # Configuration
    config, args = bert_pretraining_setup(
        CONFIG_DIR / "pretraining.yml",
        "phased",
        "base_128")

    # Setup weights and biases
    wandb_init(config, tags=['PE'], disable=args.wandb == 'False')

    # Setup dataset
    if not config.data.input_files:
        raise ValueError(f"Input files for data needs to be set.")
    dataset = load_dataset(config.data.input_files)

    # Create the training session
    session = pretraining_phased(config)

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
