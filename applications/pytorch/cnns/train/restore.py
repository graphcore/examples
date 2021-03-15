# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import torch
import poptorch
from train import train, convert_to_ipu_model, get_validation_function
from validate import validate_checkpoints
import os
import logging
from train import create_model_opts, get_optimizer, get_lr_scheduler
import sys
sys.path.append('..')
import models
import utils
import datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Restoring training run from a given checkpoint')
    parser.add_argument('--checkpoint-path', help="The path of the checkpoint file", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint_path)
    opts = checkpoint['opts']
    utils.Logger.setup_logging_folder(opts)

    logging.info("Loading the data")
    model_opts = create_model_opts(opts)
    train_data = datasets.get_data(opts, model_opts, train=True, async_dataloader=True)

    logging.info(f"Restore the {opts.model} model to epoch {checkpoint['epoch']} on {opts.data} dataset(Loss:{checkpoint['loss']}, train accuracy:{checkpoint['train_accuracy']})")
    model = models.get_model(opts, datasets.datasets_info[opts.data], pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()

    optimizer = get_optimizer(opts, model)
    lr_scheduler = get_lr_scheduler(opts, optimizer)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # set the LR scheduler to the correct position
    lr_scheduler.last_epoch += checkpoint["epoch"]
    training_model = convert_to_ipu_model(model, opts, optimizer)

    training_validation_func = get_validation_function(opts, model) if opts.validation_mode == "during" else None
    train(training_model, train_data, opts, lr_scheduler, range(checkpoint["epoch"]+1, opts.epoch+1), optimizer, training_validation_func)
    if opts.validation_mode == "after":
        checkpoint_folder = os.path.dirname(os.path.realpath(args.checkpoint_path))
        checkpoint_files = [os.path.join(checkpoint_folder, file_name) for file_name in os.listdir(checkpoint_folder) if file_name.endswith(".pt")]
        validate_checkpoints(checkpoint_files)
