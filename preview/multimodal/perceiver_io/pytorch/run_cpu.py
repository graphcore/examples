# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import os
import sys
from typing import Callable
from dataclasses import asdict

import wandb
import torch
import numpy as np
from datasets import load_metric
from transformers import Trainer, HfArgumentParser, TrainingArguments, PerceiverFeatureExtractor

from dataset_factory import DatasetArguments, get_dataset
from models.model_factory import ModelArguments, get_model


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def run_training(
    model,
    eval_ds,
    train_ds,
    args: TrainingArguments,
    compute_metrics: Callable,
    feature_extractor: PerceiverFeatureExtractor,
):

    print("Running training.")
    trainer = Trainer(
        args=args,
        model=model,
        eval_dataset=eval_ds,
        train_dataset=train_ds,
        data_collator=collate_fn,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )
    train_results = trainer.train()
    print(train_results)


def run_inference(model, test_ds, compute_metrics: Callable):

    print("Running inference.")
    image = test_ds[0]
    inputs = feature_extractor(image, return_tensors="pt").pixel_values
    outputs = model(inputs)
    logits = outputs.logits
    predicted_class = model.config.id2label[logits.argmax(-1).item()]
    print(f"Predicted class: {predicted_class}")


def parse_arguments():
    parser = HfArgumentParser((ModelArguments, DatasetArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()


def get_metrics():
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    return compute_metrics


if __name__ == "__main__":
    # parse arguments
    model_args, dataset_args, training_args = parse_arguments()
    # init wandb
    if "wandb" in training_args.report_to:
        wandb.init(
            entity="sw-apps",
            project="perceiver-imgclass",
            config=dict(**asdict(model_args), **asdict(dataset_args), **asdict(training_args)),
        )
    # get model
    model, feature_extractor = get_model(model_args)
    # get metrics
    metrics_fn = get_metrics()
    # run training
    if training_args.do_train:
        print("Preparing training dataset.")
        train_ds = get_dataset("train", dataset_args, feature_extractor)
        eval_ds = get_dataset("eval", dataset_args, feature_extractor) if training_args.do_eval else None

        run_training(
            model,
            args=training_args,
            eval_ds=eval_ds,
            train_ds=train_ds,
            compute_metrics=metrics_fn,
            feature_extractor=feature_extractor,
        )

    if training_args.do_predict:
        test_ds = get_dataset("test", dataset_args, feature_extractor)
        run_inference(model, test_ds, metrics_fn)
