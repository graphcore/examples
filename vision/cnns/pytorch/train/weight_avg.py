# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import logging
import torch
import argparse
import os
from torch.optim.swa_utils import AveragedModel
import import_helper
import models
import datasets


def load_model(checkpoint_file, model=None):
    """Load a checkpoint
    Parameters:
        checkpoint_file: the path of the checkpoint
        model: the model instance, if not defined, new instance is created"""
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint["args"]
    if model is None:
        model = models.get_model(args, datasets.datasets_info[args.data], pretrained=False, with_loss=True)
    models.load_model_state_dict(model, checkpoint["model_state_dict"])
    model.double()
    return model


def average_model_weights(checkpoint_input_path, checkpoint_output_path, average_fn, checkpoint_N):
    checkpoint_files = [
        os.path.join(checkpoint_input_path, file_name)
        for file_name in os.listdir(checkpoint_input_path)
        if file_name.endswith(".pt")
    ]

    def ckpt_key(ckpt):
        return int(ckpt.split("_")[-1].split(".")[0])

    try:
        checkpoint_files = sorted(checkpoint_files, key=ckpt_key)
    except:
        logging.warn("Checkpoint names are changed, which may cause inconsistent order.")

    # Select the last N checkpoint
    if checkpoint_N > 0 and checkpoint_N <= len(checkpoint_files):
        checkpoint_files = checkpoint_files[-checkpoint_N:]

    # initialize averaged model with first checkpoint
    model = load_model(checkpoint_files[0])
    averaged_model = AveragedModel(model, avg_fn=average_fn)

    # loop through the remaining checkpoints and update averaged model
    for checkpoint in checkpoint_files:
        model = load_model(checkpoint, model)
        averaged_model.update_parameters(model)

    last_checkpoint = torch.load(checkpoint_files[-1])
    args = last_checkpoint["args"]
    filename = f'{args.model}_{args.data}_{last_checkpoint["epoch"]}_averaged.pt'
    save_path = os.path.join(checkpoint_output_path, filename)

    if args.precision[-3:] == ".16":
        model.half()
    else:
        model.float()

    torch.save(
        {
            "epoch": last_checkpoint["epoch"] + 1,
            "model_state_dict": models.get_model_state_dict(averaged_model.module),
            "loss": 0,  # dummy just to work with validate script
            "train_accuracy": 0,  # dummy just to work with validate script
            "args": args,
        },
        save_path,
    )

    return averaged_model


def create_average_fn(args):
    if args.weight_avg_strategy == "exponential":
        return (
            lambda averaged_model_parameter, model_parameter, num_averaged: args.weight_avg_exp_decay
            * averaged_model_parameter
            + (1 - args.weight_avg_exp_decay) * model_parameter
        )
    else:  # mean strategy
        return None


def add_parser_arguments(parser):
    parser.add_argument(
        "--weight-avg-strategy", default="none", choices=["mean", "exponential", "none"], help="Weight average strategy"
    )
    parser.add_argument(
        "--weight-avg-exp-decay",
        type=float,
        default=0.99,
        help="The exponential decay constant, applied if exponential weight average strategy is chosen",
    )
    parser.add_argument(
        "--weight-avg-N",
        type=int,
        default=-1,
        help="Weight average applied on last N checkpoint, -1 means all checkpoints",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-input-path", type=str, required=True)
    parser.add_argument("--checkpoint-output-path", type=str, required=True)
    add_parser_arguments(parser)
    args = parser.parse_args()

    if args.weight_avg_strategy != "none":
        average_fn = create_average_fn(args)
        averaged_model = average_model_weights(
            args.checkpoint_input_path, args.checkpoint_input_path, average_fn, args.weight_avg_N
        )
