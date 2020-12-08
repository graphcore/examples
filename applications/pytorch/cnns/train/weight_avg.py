# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import argparse
import poptorch
import os
import sys
from torch.optim.swa_utils import AveragedModel
sys.path.append('..')
import models
import data


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    opts = checkpoint['opts']
    model = models.get_model(opts, data.datasets_info[opts.data], pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.double()
    return model


def average_model_weights(checkpoint_path, average_fn):
    checkpoint_files = [os.path.join(checkpoint_path, file_name) for file_name in os.listdir(checkpoint_path) if file_name.endswith(".pt")]

    # initialize averaged model with first checkpoint
    model = load_model(checkpoint_files[0])
    averaged_model = AveragedModel(model, avg_fn=average_fn)

    # loop through the remaining checkpoints and update averaged model
    for checkpoint in checkpoint_files:
        model = load_model(checkpoint)
        averaged_model.update_parameters(model)

    last_checkpoint = torch.load(checkpoint_files[-1])
    opts = last_checkpoint['opts']
    filename = f'{opts.model}_{opts.data}_{last_checkpoint["epoch"]}_averaged.pt'
    save_path = os.path.join(checkpoint_path, filename)

    if opts.precision[-3:] == ".16":
        model.half()
    else:
        model.float()

    torch.save({
                'epoch': last_checkpoint['epoch'] + 1,
                'model_state_dict': averaged_model.module.state_dict(),
                'loss': 0,  # dummy just to work with validate script
                'train_accuracy': 0,  # dummy just to work with validate script
                'opts': opts
                }, save_path)

    return averaged_model


def create_average_fn(opts):
    if opts.weight_avg_strategy == 'exponential':
        return lambda averaged_model_parameter, model_parameter, num_averaged:\
            opts.weight_exp_decay * averaged_model_parameter + (1-opts.weight_exp_decay) * model_parameter
    else:  # mean strategy
        return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--weight-avg-strategy', default='none', choices=['mean', 'exponential', 'none'])
    parser.add_argument('--weight-avg-exp-decay', type=float, default=0.99)

    args = parser.parse_args()

    if args.weight_avg_strategy != 'none':
        average_fn = create_average_fn(args)

        averaged_model = average_model_weights(args.checkpoint_path, average_fn)
