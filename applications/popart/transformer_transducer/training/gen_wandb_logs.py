# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

"""
Upload training results to W&B from specified folder
"""

import argparse
import wandb
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--start_step', type=int, default=0,
                        help="Starting step of run")
    args = parser.parse_args()

    wandb.init(entity="ml-perf-training", project="rnn-t", name=args.run_name)

    loss_data_fp = os.path.join(args.model_dir, 'rnnt_losses.txt')
    learning_rates_data_fp = os.path.join(args.model_dir, 'learning_rates.txt')

    with open(loss_data_fp, 'r') as f:
        rnnt_loss_data = [float(x) for x in f.readlines()]

    with open(learning_rates_data_fp, 'r') as f:
        learning_rate_data = [float(x) for x in f.readlines()]

    for step, (loss, lr) in enumerate(zip(rnnt_loss_data, learning_rate_data)):
        wandb.log({"RNN-T Loss": loss, "Learning Rate": lr, 'custom_step': args.start_step + step})
