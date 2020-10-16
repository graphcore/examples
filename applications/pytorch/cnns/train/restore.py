# Copyright 2020 Graphcore Ltd.
import argparse
import torch
from train import train, convert_to_ipu_model
from data import get_data, datasets_info
from validate import validate_checkpoints
import os
import logging
from train import create_model_opts, get_optimizer
import sys
sys.path.append('..')
import models  # noqa: E402

# Set up logging
logging.basicConfig(format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Restoring training run from a given checkpoint')
    parser.add_argument('--checkpoint-path', help="The path of the checkpoint file", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint_path)
    opts = checkpoint['opts']

    logging.info("Loading the data")
    model_opts = create_model_opts(opts)
    train_data, test_data = get_data(opts, model_opts)

    logging.info("Restore the {0} model to epoch {1} on {2} dataset(Loss:{3}, train accuracy:{4})".format(opts.model, checkpoint["epoch"], opts.data, checkpoint["loss"], checkpoint["train_accuracy"]))
    model = models.get_model(opts, datasets_info[opts.data], pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()

    optimizer, lr_scheduler = get_optimizer(opts, model)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    training_model = convert_to_ipu_model(model, opts, optimizer)


    train(training_model, train_data, opts, lr_scheduler, range(checkpoint["epoch"]+1, opts.epoch+1), optimizer)

    checkpoint_folder = os.path.dirname(os.path.realpath(args.checkpoint_path))
    checkpoint_files = [os.path.join(checkpoint_folder, file_name) for file_name in os.listdir(checkpoint_folder)]
    validate_checkpoints(checkpoint_files, test_data=test_data)
