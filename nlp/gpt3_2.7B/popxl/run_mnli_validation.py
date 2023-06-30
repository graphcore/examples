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
import popxl_addons as addons

from mnli_inference import mnli_inference
from mnli_finetuning import mnli_finetuning
from config import CONFIG_DIR, GPTConfig
from utils.setup import gpt_config_setup
from modelling.hf_mapping import hf_mapping_lm_to_class_inference_TP
from utils.utils import linear_schedule


def validation(config: GPTConfig, dataset, session: addons.TaskSession):
    tp = config.execution.tensor_parallel
    rf = session.ir.instance_replication_factor

    samples_per_step = config.execution.device_iterations * config.execution.micro_batch_size

    val_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=samples_per_step,
        shuffle=False,
        drop_last=True,
        collate_fn=concat_fnc,  # By default DataLoader stacks batches vertically instead of horizontally
    )

    predictions_all = []
    labels_all = []

    logging.info("Validating...")
    with session:
        for data in tqdm(val_dl):
            data_map = {}
            words = data["input_ids"]
            unpadded_length = data["unpadded_length"]
            labels = data["label"]

            words = to_numpy(np.concatenate(words), session.inputs.words.dtype, copy=False).reshape(
                -1, *session.inputs.words.shape
            )
            unpadded_length = to_numpy(unpadded_length, session.inputs.unpadded_length.dtype, copy=False).reshape(
                -1, *session.inputs.unpadded_length.shape
            )
            labels = to_numpy(labels, copy=False).flatten()

            data_map = {}
            data_map[session.inputs.words] = tensor_parallel_input(words, tp, rf)
            data_map[session.inputs.unpadded_length] = tensor_parallel_input(unpadded_length, tp, rf)

            outputs = session.run(data_map)
            logits = tensor_parallel_output(
                outputs[session.outputs["logits"]],
                session.ir.num_host_transfers,
                tp,
                rf,
                session.outputs["logits"].shape,
                tp_identical=True,
            )

            predictions = np.argmax(logits, axis=-1)

            predictions_all += list(predictions.flatten())
            labels_all += list(labels)

    metric = evaluate.load("glue", "mnli")
    metrics = metric.compute(predictions=predictions_all, references=labels_all)
    logging.info(metrics)
    for k, v in metrics.items():  # type: ignore
        wandb.run.summary["validation_" + k] = v

    return session


def main():
    # Configuration
    config, args, pretrained = gpt_config_setup(
        CONFIG_DIR / "mnli_finetuning.yml",
        "release",
        "gpt2_small",
        wandb_setup=True,
        hf_model_setup=False,
    )

    if config.checkpoint.load is None:
        raise ValueError("You must provide `config.checkpoint.load`")

    # Setup dataset
    train_dataset = prepare_dataset(
        config, "train"
    )  # There is currently an issue with the caching mechanism and it is required to load this first
    validation_dataset = prepare_dataset(config, "validation_matched")

    # Validation session
    logging.info("Compiling Validation Graph.")
    config.model.eval = True
    config.execution.micro_batch_size = 16
    config.execution.data_parallel = 1
    logging.info("Validation config")
    logging.info(config)
    val_session = mnli_inference(config)
    val_session.load_checkpoint(config.checkpoint.load)

    # Validation
    val_session = validation(config, validation_dataset, val_session)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e, exc_info=False)  # Log time of exception
        raise
