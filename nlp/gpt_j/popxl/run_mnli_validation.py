#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import logging
from functools import partial, reduce
from glob import glob
import re
import os

import torch
from tqdm import tqdm
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
import wandb

from popxl.utils import to_numpy

from inference import inference
from modelling.embedding import GPTJEmbeddingsTP
from popxl_addons import TaskSession, timer
from utils.setup import gptj_config_setup
from utils.utils import tensor_parallel_input, repeat
from utils.inference import batch_inference
from data.mnli_data import form_text, prepare_validation_features, split_text, postprocess_mnli_predictions
from config import GPTJConfig

OUTPUT_LENGTH = 5


def unwrap(dl):
    for example in tqdm(dl):
        yield torch.tensor(example["input_ids"], dtype=torch.long)


def run_validation(config: GPTJConfig, session: TaskSession, checkpoint_path: str, dataset, tokenizer, labels):
    tp = config.execution.tensor_parallel
    rf = config.execution.tensor_parallel * config.execution.data_parallel

    with timer('Loading pretrained checkpoint from file to IPU'):
        with session:
            session.load_checkpoint(checkpoint_path)

    def next_token(inputs, lengths):
        data_map = {}
        words = to_numpy(inputs, session.inputs.words.dtype).reshape(
            -1, *session.inputs.words.shape)
        data_map[session.inputs.words] = tensor_parallel_input(
            words, tp, rf, partial(GPTJEmbeddingsTP.offset_input,
                                   config=config)).squeeze()
        data_map[session.inputs.last_token_indices] = repeat(lengths - 1,
                                                             tp,
                                                             axis=0)
        # identical for all tp, take first
        next_token_id = session.run(data_map)[session.outputs.next_token][0]
        return torch.LongTensor(next_token_id)

    with timer('Running validation'):
        with session:
            answers = batch_inference(
                unwrap(dataset),
                next_token,
                config.model.sequence_length,
                eos_token_id=tokenizer.eos_token_id,  # index 50256
                pad_token_id=tokenizer.pad_token_id,  # index 50257
                output_length=OUTPUT_LENGTH,
                micro_batch_size=config.execution.micro_batch_size)

    logging.info("Computing validation metric")
    answers = [tokenizer.decode(a, skip_special_tokens=True) for a in answers]
    metric = load_metric('glue', 'mnli_mismatched')

    labels = postprocess_mnli_predictions(labels)
    formatted_answers = postprocess_mnli_predictions(answers)
    metrics = metric.compute(predictions=formatted_answers, references=labels)
    logging.info(metrics)
    return metrics


def main():
    # --- Config ---
    config, args, _ = gptj_config_setup(
        'config/inference.yml', 'release', 'gpt-j-mnli', hf_model_setup=False, wandb_setup=True)
    assert config.checkpoint.load is not None, 'You must specify a checkpoint to load using --load'

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
    tokenizer.add_special_tokens({'pad_token':
                                  '<|extratoken_1|>'})  # index 50257

    # --- Dataset ---
    dataset = load_dataset('glue', 'mnli', split='validation_mismatched')
    dataset = dataset.map(
        form_text, remove_columns=['hypothesis', 'premise', 'label', 'idx'])
    dataset = dataset.map(split_text)
    labels = dataset['class_label']
    dataset = dataset.map(prepare_validation_features,
                          batched=True,
                          remove_columns=dataset.column_names,
                          load_from_cache_file=True,
                          fn_kwargs={"tokenizer": tokenizer})
    max_len = reduce(lambda l, e: max(l, len(e["input_ids"])), dataset, 0)
    config.model.sequence_length = max_len + OUTPUT_LENGTH

    # --- Model ---
    session = inference(config)

    # --- Validation ---
    # Parse step from filename
    files = glob(os.path.expanduser(config.checkpoint.load))
    for i, f in enumerate(files):
        step = re.match(r'.*step_(\d+).*', f)
        step = int(step.groups()[0]) if step and len(step.groups()) else -1
        files[i] = step, f
    files = sorted(files)

    for step, f in files:
        step = step if step > 0 else None
        logging.info(
            f'Starting validation. File: {f}. Step: {step or "Not known"}')
        metrics = run_validation(
            config, session, f, dataset, tokenizer, labels)

        if args.wandb:
            for k, v in metrics.items():  # type: ignore
                wandb.log({k: v, 'file': f}, step=step)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception(e)  # Log time of exception
        raise
