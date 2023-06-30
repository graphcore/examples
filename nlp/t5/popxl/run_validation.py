#!/usr/bin/env python3
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import logging
from functools import partial, reduce
from glob import glob
import re
import os
from typing import Optional

import torch
from tqdm import tqdm
from datasets import load_metric

import wandb

from popxl.utils import to_numpy

from inference import inference
from modelling.embedding import T5EmbeddingsTP
from popxl_addons import TaskSession, timer
from popxl_addons.array_munging import repeat, tensor_parallel_input
from utils.setup import t5_config_setup
from utils.inference import batch_inference
from data.mnli_data import prepare_validation_dataset, postprocess_mnli_predictions
from config import T5Config, CONFIG_DIR


def unwrap(dl):
    for example in tqdm(dl):
        yield (
            torch.tensor(example["input_ids"], dtype=torch.long),
            torch.tensor(example["attention_mask"], dtype=torch.float16),
        )


def run_validation(
    config: T5Config, session: TaskSession, dataset, tokenizer, trained_session: Optional[TaskSession] = None
):
    """
    The session must be opened before calling run_validation
    Usage:
        with session:
            run_validation(config, session, dataset, tokenizer, labels, checkpoint_path)
    """
    if config.checkpoint.load:
        session.load_checkpoint(config.checkpoint.load, "none")
    elif trained_session:
        session.load_from_session(trained_session)

    # else, it assumes weights have been loaded

    tp = config.execution.tensor_parallel
    rf = config.execution.tensor_parallel * config.execution.data_parallel

    def next_token(enc_batch, enc_masks, dec_batch, dec_masks, batch_lens):
        data_map = {}
        words = to_numpy(enc_batch, session.inputs.words.dtype).reshape(-1, *session.inputs.words.shape)
        attention_mask = to_numpy(enc_masks, session.inputs.attention_mask.dtype).reshape(
            -1, *session.inputs.attention_mask.shape
        )
        decoder_words = to_numpy(dec_batch, session.inputs.decoder_words.dtype).reshape(
            -1, *session.inputs.decoder_words.shape
        )
        decoder_attention_mask = to_numpy(dec_masks, session.inputs.decoder_attention_mask.dtype).reshape(
            -1, *session.inputs.decoder_attention_mask.shape
        )
        data_map[session.inputs.words] = tensor_parallel_input(
            words, tp, rf, partial(T5EmbeddingsTP.offset_input, config=config)
        )
        data_map[session.inputs.attention_mask] = tensor_parallel_input(attention_mask, tp, rf)
        data_map[session.inputs.decoder_words] = tensor_parallel_input(
            decoder_words, tp, rf, partial(T5EmbeddingsTP.offset_input, config=config)
        )
        data_map[session.inputs.decoder_attention_mask] = tensor_parallel_input(decoder_attention_mask, tp, rf)
        data_map[session.inputs.last_token_indices] = repeat(batch_lens - 1, tp, axis=0)
        # identical for all tp, take first
        next_token_id = session.run(data_map)[session.outputs.next_token][0]
        return torch.LongTensor(next_token_id)

    with timer("Running validation"):
        answers = batch_inference(
            unwrap(dataset),
            next_token,
            config.model.sequence_length,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            output_length=config.inference.output_length,
            micro_batch_size=config.execution.micro_batch_size,
        )

    logging.info("Computing validation metric")
    answers = [tokenizer.decode(a, skip_special_tokens=True) for a in answers]
    return answers


def main():
    # --- Config ---
    config, args, pretrained = t5_config_setup(
        CONFIG_DIR / "inference.yml", "release", "xxl-mnli", hf_model_setup=False, wandb_setup=True
    )
    assert config.checkpoint.load is not None, "You must specify a checkpoint to load using --load"

    # --- Dataset ---
    with timer("Data preparation"):
        dataset, tokenizer = prepare_validation_dataset(config)
        labels = dataset["label"]

    # --- Metric ---
    metric = load_metric("glue", "mnli_mismatched")

    # --- Model ---
    max_len = reduce(lambda l, e: max(l, sum(e["attention_mask"])), dataset, 0)
    config.model.sequence_length = max_len
    logging.info(f"Reducing sequence length to {max_len}")

    session = inference(config)

    files = glob(os.path.expanduser(config.checkpoint.load))
    for i, f in enumerate(files):
        step = re.match(r".*step_(\d+).*", f)
        step = int(step.groups()[0]) if step and len(step.groups()) else -1
        files[i] = step, f
    files = sorted(files)

    for step, f in files:
        step = step if step > 0 else None
        logging.info(f'Starting validation. File: {f}. Step: {step or "Not known"}')
        config.checkpoint.load = f
        with session:
            answers = run_validation(config, session, dataset, tokenizer)
            formatted_answers = postprocess_mnli_predictions(answers)
            metrics = metric.compute(predictions=formatted_answers, references=labels)
            logging.info(metrics)

        if args.wandb:
            for k, v in metrics.items():
                wandb.log({k: v}, step=step)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)  # Log time of exception
        raise
