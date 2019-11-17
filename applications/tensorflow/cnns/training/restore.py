# Copyright 2019 Graphcore Ltd.
import argparse
import json
import os
import importlib
import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Restoring training run from a checkpoint')
    parser.add_argument('--restore-path', help="Log folder to restore from", required=True)
    args = parser.parse_args()
    args = vars(args)
    with open(os.path.join(args["restore_path"], 'arguments.json'), 'r') as fp:
        opts = json.load(fp)

    print(opts)

    opts['restoring'] = True

    try:
        model = importlib.import_module("Models." + opts['model'])
    except ImportError:
        raise ValueError('Models/{}.py not found'.format(opts['model']))

    try:
        lr_schedule = importlib.import_module("LR_Schedules." + opts['lr_schedule'])
    except ImportError:
        raise ValueError("LR_Schedules/{}.py not found".format(opts['lr_schedule']))

    train.train_process(model.Model, lr_schedule.LearningRate, opts)
