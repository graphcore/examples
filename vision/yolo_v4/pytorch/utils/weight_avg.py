# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import Callable
from models.yolov4_p5 import Yolov4P5
from torch.optim.swa_utils import AveragedModel
import utils.tools
import pathlib
import torch
from yacs.config import CfgNode
import re


def average_model_weights(cfg: CfgNode) -> torch.nn.Module:

    # list checkpoints
    checkpoints_folder = pathlib.Path(cfg.training.checkpoint_dir, 'weights')
    checkpoints = list(checkpoints_folder.glob('*.pt'))

    def sorting_fn(ckpt_filename):
        match = re.match(r'train_epoch_(\d+)_of_\d+\.pt', str(ckpt_filename))
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f'Found checkpoint filename with unexpected format: {ckpt_filename}')

    checkpoints = sorted(checkpoints, key=sorting_fn)


    if len(checkpoints) > 1:

        print(f'Performing weight averaging on the following checkpoints: {checkpoints}, with decay factor {cfg.training.weight_avg_decay}')

        # build model
        model = Yolov4P5(cfg)

        # create model that contains the average of the weights
        average_fn = create_average_fn(cfg.training.weight_avg_decay)
        averaged_model = AveragedModel(model, avg_fn=average_fn)

        # loop through the remaining checkpoints and update averaged model
        for checkpoint in checkpoints:
            load_model(model, checkpoint)
            averaged_model.update_parameters(model)

        return averaged_model.module
    else:
        raise ValueError(f'Not enough model checkpoints to perform weight averaging. Found {len(checkpoints)}, expected at least 2.')


def load_model(model: torch.nn.Module, checkpoint: str) -> None:
    weights = dict(utils.tools.load_weights(checkpoint))
    model.load_state_dict(weights)


def create_average_fn(weight_avg_exp_decay: float) -> Callable:
    return lambda averaged_model_parameter, model_parameter, num_averaged:\
            weight_avg_exp_decay * averaged_model_parameter + (1-weight_avg_exp_decay) * model_parameter
