# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

"""
Upload training results to W&B from specified folder
"""

import argparse
import wandb
import checkpoint_utils
import os


def init_wandb(entity, run_name):
    wandb.init(entity=entity,
               project="rnn-t", name=run_name)


def log_wandb_data(step, loss, learning_rate, step_time, throughput, wer=None):
    wandb.log({"RNN-T loss": loss, "Learning rate": learning_rate,
              "Step": step, "Step time": step_time, "Throughput": throughput})
    if wer is not None:
        wandb.log({"WER": wer})


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument("--wandb-entity", type=str, default="ml-perf-training")
    parser.add_argument('--wandb-run-name', type=str, required=True)
    parser.add_argument('--start-step', type=int, default=0,
                        help="Starting step of run")
    args = parser.parse_args()

    init_wandb(args.wandb_entity, args.wandb_run_name)

    loss_data_fp = checkpoint_utils.get_data_fp(args.model_dir, "loss")
    learning_rates_data_fp = checkpoint_utils.get_data_fp(
        args.model_dir, 'learning_rate')
    step_time_data_fp = checkpoint_utils.get_data_fp(
        args.model_dir, "step_time")
    throughput_data_fp = checkpoint_utils.get_data_fp(
        args.model_dir, "throughput")
    wer_data_fp = checkpoint_utils.get_data_fp(args.model_dir, "wer")

    with open(loss_data_fp, 'r') as f:
        rnnt_loss_data = [float(x) for x in f.readlines()]

    with open(learning_rates_data_fp, 'r') as f:
        learning_rate_data = [float(x) for x in f.readlines()]

    with open(step_time_data_fp, 'r') as f:
        step_time_data = [float(x) for x in f.readlines()]

    with open(throughput_data_fp, 'r') as f:
        throughput_data = [float(x) for x in f.readlines()]

    # wer logging only if --do-validation with training
    wer_data = None
    wer = None
    if os.path.exists(wer_data_fp):
        wer_data = {}
        with open(wer_data_fp, 'r') as f:
            for line in f.readlines():
                step, score = line.split(",")
                wer_data[int(step.strip())] = float(score.strip())

    for step, (loss, lr, step_time, throughput) in enumerate(zip(rnnt_loss_data, learning_rate_data, step_time_data, throughput_data)):
        step += args.start_step
        if (wer_data is not None) and (step in wer_data):
            wer = wer_data[step]
        else:
            wer = None
        log_wandb_data(step, loss, lr, step_time, throughput, wer)
