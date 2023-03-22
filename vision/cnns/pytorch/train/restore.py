# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import torch
import os
import logging
from train import (
    train,
    convert_to_ipu_model,
    get_validation_function,
    create_training_opts,
    get_optimizer,
    get_lr_scheduler,
)
from validate import validate_checkpoints
import import_helper
import models
import utils
import datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restoring training run from a given checkpoint")
    parser.add_argument("--checkpoint-input-path", help="The path of the checkpoint file to load", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint_input_path)
    args = checkpoint["args"]
    utils.Logger.setup_logging_folder(args)

    opts = create_training_opts(args)
    train_data = datasets.get_data(args, opts, train=True, async_dataloader=True)

    logging.info(
        f"Restore the {args.model} model to epoch {checkpoint['epoch']} on {args.data} dataset(Loss:{checkpoint['loss']}, train accuracy:{checkpoint['train_accuracy']})"
    )
    model = models.get_model(
        args,
        datasets.datasets_info[args.data],
        pretrained=False,
        use_mixup=args.mixup_enabled,
        use_cutmix=args.cutmix_enabled,
        with_loss=True,
        inference_mode=False,
    )
    models.load_model_state_dict(model, checkpoint["model_state_dict"])

    optimizer = get_optimizer(args, model)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler = get_lr_scheduler(args, optimizer, len(train_data), start_epoch=checkpoint["epoch"])
    training_model = convert_to_ipu_model(model, args, optimizer)

    if args.validation_mode == "during":
        training_validation_func = get_validation_function(args, model).func
    else:
        training_validation_func = None

    train(
        training_model,
        train_data,
        args,
        lr_scheduler,
        range(checkpoint["epoch"] + 1, args.epoch + 1),
        optimizer,
        training_validation_func,
    )

    if args.validation_mode == "after":
        checkpoint_folder = os.path.dirname(os.path.realpath(args.checkpoint_input_path))
        checkpoint_files = [
            os.path.join(checkpoint_folder, file_name)
            for file_name in os.listdir(checkpoint_folder)
            if file_name.endswith(".pt")
        ]
        validate_checkpoints(checkpoint_files)
