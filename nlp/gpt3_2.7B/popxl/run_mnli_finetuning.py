#!/usr/bin/env python3
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from copy import deepcopy
import logging
import time

import wandb
import torch
import numpy as np
from tqdm import tqdm
import evaluate

import popxl
from popxl.utils import to_numpy
from popxl_addons.array_munging import tensor_parallel_input, tensor_parallel_output
from data.mnli.mnli_data import prepare_dataset, concat_fnc
from popxl_addons import timer

from mnli_inference import mnli_inference
from mnli_finetuning import mnli_finetuning
from config import CONFIG_DIR, GPTConfig
from utils.setup import gpt_training_setup
from modelling.hf_mapping import hf_mapping_lm_to_class_inference_TP
from utils.utils import linear_schedule
from run_mnli_validation import validation


def training(config: GPTConfig, dataset, pretrained):
    # Build and compile program
    logging.info("Compiling Training Graph.")
    session = mnli_finetuning(config)

    tp = config.execution.tensor_parallel
    rf = session.ir.instance_replication_factor

    # Load checkpoint or pretrained
    if config.checkpoint.load is not None:
        with timer("Loading pretrained checkpoint from file to IPU"):
            session.load_checkpoint(config.checkpoint.load)
    elif pretrained:
        with timer("Loading HF pretrained model to IPU"):
            session.write_variables_data(hf_mapping_lm_to_class_inference_TP(config, session, pretrained))
    else:
        logging.info(f"Not loading a pretrained model.")

    samples_per_step = config.execution.device_iterations * config.training.global_batch_size

    train_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=samples_per_step,
        shuffle=True,
        drop_last=True,
        collate_fn=concat_fnc,  # By default DataLoader stacks batches vertically instead of horizontally
    )

    step = 0
    total_steps = config.training.epochs * len(train_dl)
    logging.info(f"Total steps: {total_steps}")

    if config.training.optimizer.learning_rate.constant:
        lr_schedule = {i: config.training.optimizer.learning_rate.maximum for i in range(total_steps + 1)}
    else:
        lr_schedule = linear_schedule(
            total_steps,
            1e-7,
            config.training.optimizer.learning_rate.maximum,
            config.training.optimizer.learning_rate.warmup_proportion,
        )

    metric = evaluate.load("glue", "mnli")

    # Training loop
    with session:
        start = time.perf_counter()
        for epoch in range(config.training.epochs):
            for data in train_dl:
                words = data["input_ids"]
                unpadded_length = data["unpadded_length"]
                labels = data["label"]

                words = to_numpy(words, session.inputs.words.dtype, copy=False).reshape(-1, *session.inputs.words.shape)
                unpadded_length = to_numpy(unpadded_length, session.inputs.unpadded_length.dtype, copy=False).reshape(
                    -1, *session.inputs.unpadded_length.shape
                )
                labels = to_numpy(labels, session.inputs.labels.dtype, copy=False).reshape(
                    -1, *session.inputs.labels.shape
                )
                lr = np.full((session.ir.num_host_transfers, rf), lr_schedule[step]).astype("float32").squeeze()

                data_map = {}
                data_map[session.inputs.words] = tensor_parallel_input(words, tp, rf)
                data_map[session.inputs.unpadded_length] = tensor_parallel_input(unpadded_length, tp, rf)
                data_map[session.inputs.labels] = tensor_parallel_input(labels, tp, rf)
                data_map[session.inputs.lr] = lr

                # Run program
                outputs = session.run(data_map)  # type: ignore
                losses = tensor_parallel_output(
                    outputs[session.outputs["loss"]],
                    session.ir.num_host_transfers,
                    tp,
                    rf,
                    session.outputs["loss"].shape,
                    tp_identical=True,
                )
                logits = tensor_parallel_output(
                    outputs[session.outputs["logits"]],
                    session.ir.num_host_transfers,
                    tp,
                    rf,
                    session.outputs["logits"].shape,
                    tp_identical=True,
                )
                predictions = np.argmax(logits, axis=-1).flatten()

                accuracy = metric.compute(predictions=predictions, references=labels.flatten())["accuracy"]

                # Logging
                duration = time.perf_counter() - start
                start = time.perf_counter()

                loss = np.mean(losses.astype(np.float32))
                throughput = samples_per_step / duration
                total_steps = config.execution.device_iterations * step
                result_str = (
                    f"Epoch: {epoch} "
                    f"Step: {total_steps} "
                    f"Loss: {loss:5.3f} "
                    f"Duration: {duration:6.4f} s "
                    f"Throughput: {throughput:6.1f} samples/sec "
                    f"Accuracy: {accuracy:1.2f}"
                )
                logging.info(result_str)
                wandb.log(
                    {"Loss": loss, "LR": lr_schedule[step], "Throughput": throughput, "train_accuracy": accuracy},
                    step=total_steps,
                )
                step += 1

    return session


def main():
    # Configuration
    config, args, pretrained = gpt_training_setup(CONFIG_DIR / "mnli_finetuning.yml", "release", "gpt2_small")
    config.training.optimizer.learning_rate.constant = True

    # Setup dataset
    train_dataset = prepare_dataset(config, "train")
    validation_dataset = prepare_dataset(config, "validation_matched")

    # Train
    train_session = training(config, train_dataset, pretrained)

    # Validation session
    logging.info("Compiling Validation Graph.")
    config.model.eval = True
    config.execution.micro_batch_size = 16
    config.execution.data_parallel = 1
    logging.info("Validation config")
    logging.info(config)
    val_session = mnli_inference(config)
    val_session.load_from_session(train_session)

    # Validation
    validation(config, validation_dataset, val_session)

    # Save checkpoint
    if config.checkpoint.save is not None:
        val_session.save_checkpoint(config.checkpoint.save)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e, exc_info=False)  # Log time of exception
        raise
