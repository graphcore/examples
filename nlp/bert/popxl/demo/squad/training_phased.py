#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import logging
import time

import wandb
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, load_metric

import popxl
from popxl.utils import to_numpy

from config import CONFIG_DIR, BertConfig
from utils.setup import bert_fine_tuning_setup, wandb_init
from execution.task_session import hf_mapping
from execution.utils import linear_schedule
from execution.squad import squad_inference_phased, squad_training_phased
from data.squad_data import PadCollate, postprocess_qa_predictions, prepare_train_features, prepare_validation_features


def training(config: BertConfig, dataset, pretrained):
    # Build and compile program
    logging.info("Compiling Training Graph.")
    session = squad_training_phased(config)

    # Write pretrained checkpoint to the IPU
    if config.checkpoint.load:
        session.load_checkpoint(config.checkpoint.load, 'none')
    else:
        session.write_variables_data(hf_mapping(config, session, pretrained))

    dataset = dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=1,
        remove_columns=dataset.column_names,
        load_from_cache_file=True,
    )

    samples_per_step = config.execution.device_iterations * config.training.global_batch_size

    train_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=samples_per_step,
        shuffle=True,
        drop_last=False,
        collate_fn=PadCollate(samples_per_step,
                              # This is the ignore_index
                              {"labels": config.model.sequence_length}))

    step = 0

    lr_schedule = linear_schedule(
        config.training.epochs * len(train_dl),
        1e-7,
        config.training.optimizer.learning_rate.maximum,
        config.training.optimizer.learning_rate.warmup_proportion)

    # Training loop
    with session:
        start = time.perf_counter()
        for _ in range(config.training.epochs):
            for data in train_dl:
                data_map = {}
                for idx, key in enumerate(['input_ids', 'token_type_ids', 'attention_mask', 'labels']):
                    h2d = session.inputs[idx]
                    data_map[h2d] = to_numpy(data[key], h2d.dtype)\
                        .reshape(session.ir.num_host_transfers, config.execution.data_parallel, *h2d.shape)

                seeds = popxl.create_seeds(
                    config.model.seed, step, batches_per_step=session.ir.num_host_transfers, replicas=config.execution.data_parallel)
                data_map[session.inputs[4]] = seeds

                # Add learning rate inputs
                # TODO: Allow broadcasted inputs
                for h2d in session.inputs[len(data_map):]:
                    # TODO: Allow accepting of smaller sized inputs.
                    data_map[h2d] = np.full(
                        (session.ir.num_host_transfers, config.execution.data_parallel, 1), lr_schedule[step]).astype(np.float32)

                # Run program
                losses = session.run(data_map)  # type: ignore
                losses_np: np.ndarray = np.asarray([losses[d2h] for d2h in session.outputs])

                # Logging
                duration = time.perf_counter() - start
                start = time.perf_counter()

                loss = np.mean(losses_np.astype(np.float32))
                throughput = samples_per_step/duration
                total_steps = config.execution.device_iterations * step
                result_str = (
                    f"Step: {total_steps} "
                    f"Loss: {loss:5.3f} "
                    f"Duration: {duration:6.4f} s "
                    f"Throughput: {throughput:6.1f} samples/s ")
                logging.info(result_str)
                wandb.log({"Loss": loss,
                           "LR": lr_schedule[step],
                           "Throughput": throughput},
                          step=total_steps)
                step += 1

    return session


def validation(config: BertConfig, dataset, train_session):
    # Configuration
    config.execution.micro_batch_size = 16
    config.execution.data_parallel = 1

    logging.info("Compiling Validation Graph.")
    session = squad_inference_phased(config)

    session.load_from_session(train_session)

    features = dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=1,
        remove_columns=dataset.column_names,
        load_from_cache_file=True)

    samples_per_step = config.execution.device_iterations * config.execution.micro_batch_size

    val_dl = torch.utils.data.DataLoader(
        features.remove_columns(
            ['example_id', 'offset_mapping']),
        batch_size=samples_per_step,
        shuffle=False,
        drop_last=False,
        collate_fn=PadCollate(samples_per_step))

    raw_predictions = [[], []]

    logging.info("Validating...")
    with session:
        for data in tqdm(val_dl):
            data_map = {}
            for idx, key in enumerate(["input_ids", "token_type_ids", "attention_mask"]):
                h2d = session.inputs[idx]
                data_map[h2d] = to_numpy(data[key], h2d.dtype).reshape(h2d.shape)

            outputs: np.ndarray = session.run(data_map)[session.outputs[0]]  # type: ignore

            start, end = np.split(outputs.astype(np.float32), 2, axis=-1)
            raw_predictions[0].append(
                start.reshape(-1, config.model.sequence_length))
            raw_predictions[1].append(
                end.reshape(-1, config.model.sequence_length))

    raw_predictions[0] = np.concatenate(raw_predictions[0], axis=0)
    raw_predictions[1] = np.concatenate(raw_predictions[1], axis=0)

    final_predictions = postprocess_qa_predictions(dataset,
                                                   features,
                                                   raw_predictions)

    metric = load_metric("squad")
    formatted_predictions = [{"id": k, "prediction_text": v}
                             for k, v in final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]}
                  for ex in dataset]
    metrics = metric.compute(
        predictions=formatted_predictions, references=references)
    logging.info(metrics)
    for k, v in metrics.items():  # type: ignore
        wandb.run.summary[k] = v

    return session


def main():
    config, args, pretrained = bert_fine_tuning_setup(
        CONFIG_DIR / "squad_training.yml",
        "phased",
        "base",
    )

    wandb_init(config, tags=['PE'], disable=args.wandb == 'False')

    # Setup dataset
    dataset = load_dataset('squad')

    # Train
    train_session = training(config, dataset["train"], pretrained)

    # Test
    val_session = validation(config, dataset["validation"], train_session)

    # Save checkpoint
    if config.checkpoint.save is not None:
        val_session.save_checkpoint(config.checkpoint.save)


if __name__ == "__main__":
    main()
