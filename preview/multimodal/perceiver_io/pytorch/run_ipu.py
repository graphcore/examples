# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import torch
import wandb
import numpy as np
from dataclasses import asdict

from datasets import load_metric
from optimum.graphcore import IPUConfig
import timm

from parsing import parse_arguments
from dataset_factory import get_dataset
from models.model_factory import get_model
from perceiver_trainer import PerceiverTrainer


import logging

logging.basicConfig(filename="std.log", format="%(asctime)s %(message)s", level=logging.DEBUG, filemode="w")


class MixupCollateFn(timm.data.Mixup):
    def __call__(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        pixel_values, labels = super().__call__(pixel_values, labels)
        return {"pixel_values": pixel_values, "labels": labels}


def default_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def get_collate_fn(mixup_alpha, cutmix_alpha, num_classes):

    if mixup_alpha > 0.0 or cutmix_alpha > 0.0:
        return MixupCollateFn(mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, num_classes=num_classes)
    else:
        return default_collate_fn


def get_metrics():
    metric = load_metric("accuracy")

    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

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
    print("Preparing training dataset.")
    train_ds = get_dataset("train", dataset_args, feature_extractor)
    eval_ds = get_dataset("eval", dataset_args, feature_extractor) if training_args.do_eval else None
    # set up IPUConfig
    ipu_config = IPUConfig.from_pretrained(training_args.ipu_config_name)
    ipu_config.profile_dir = training_args.profile_dir if training_args.profile_dir != "" else ipu_config.profile_dir
    collate_fn = get_collate_fn(
        mixup_alpha=dataset_args.mixup_alpha, cutmix_alpha=dataset_args.cutmix_alpha, num_classes=model_args.num_labels
    )

    print("Running training.")
    trainer = PerceiverTrainer(
        model=model,
        args=training_args,
        ipu_config=ipu_config,
        eval_dataset=eval_ds,
        train_dataset=train_ds,
        data_collator=collate_fn,
        tokenizer=feature_extractor,
        compute_metrics=metrics_fn,
    )
    train_results = trainer.train()
    print(train_results)
