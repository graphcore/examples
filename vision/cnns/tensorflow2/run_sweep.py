# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import argparse
import os
import sys
import subprocess


def add_poprun_arguments(parser):
    # shared application arguments
    parser.add_argument("--config", default=None, type=str, required=True)
    parser.add_argument("--num_epochs", default=None, type=int, required=True)

    parser.add_argument("--init_lr", default=None, type=float, required=True)
    parser.add_argument("--end_lr_ratio", default=None, type=float, required=True)
    parser.add_argument("--warmup_epochs", default=None, type=int, required=True)
    parser.add_argument("--bn_momentum", default=None, type=float, required=True)
    parser.add_argument("--label_smoothing", default=None, type=float, required=True)
    parser.add_argument("--opt_momentum", default=None, type=float, required=True)
    # SGD specific
    parser.add_argument("--l2", default=None, type=float, required=False)
    # LARS specific
    parser.add_argument("--lars_weight_decay", default=None, type=float, required=False)
    parser.add_argument("--lars_eeta", default=None, type=float, required=False)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_poprun_arguments(parser)
    args = parser.parse_args()

    hosts = os.environ.get("POPRUN_HOSTS")
    vipu_host = os.environ.get("IPUOF_VIPU_API_HOST")
    vipu_partition = os.environ.get("IPUOF_VIPU_API_PARTITION_ID")
    if None in [hosts, vipu_host, vipu_partition]:
        raise ValueError(
            f"The following environment variables must be defined: "
            f"POPRUN_HOSTS={hosts}\n"
            f"IPUOF_VIPU_API_HOST={vipu_host}\n"
            f"IPUOF_VIPU_API_PARTITION_ID={vipu_partition}"
        )

    user = os.environ["USER"]
    exec_cache = os.environ.get("TF_POPLAR_EXEC_CACHE") or os.path.join("/home", user, "exec_cache")

    poprun_command = [
        "poprun",
        "-vv",
        "--host",
        hosts,
        "--only-output-from-instance",
        "0",
        "--mpi-global-args",
        "--mca oob_tcp_if_include eno1 --mca btl_tcp_if_include eno1",
        "--update-partition",
        "yes",
        "--reset-partition",
        "no",
        "--vipu-server-timeout",
        "600",
        "--vipu-server-host",
        vipu_host,
        "--vipu-partition",
        vipu_partition,
        "--executable-cache-path",
        exec_cache,
        "--num-instances",
        64,
        "--num-replicas",
        64,
        "--ipus-per-replica",
        1,
    ]

    training_command = [
        *poprun_command,
        "python3",
        "train.py",
        "--config",
        args.config,
        "--num-epochs",
        args.num_epochs,
        "--wandb",
        "True",
        "--target-accuracy",
        0.759,
        "--ckpts-per-epoch",
        1,
        "--first-ckpt-epoch",
        0,
        "--sweep",
        "True",
    ]

    # label smoothing
    training_command += ["--label-smoothing", args.label_smoothing]
    # l2 regularization
    training_command += ["--l2-regularization", args.l2] if args.l2 is not None else []
    # norm layer
    training_command += ["--norm-layer", "{" + f'"name": "custom_batch_norm", "momentum": {args.bn_momentum}' + "}"]
    # lr schedule params
    lr_schedule_params = [
        f'"initial_learning_rate": {args.init_lr}',
        f'"end_learning_rate_ratio": {args.end_lr_ratio}',
        f'"epochs_to_total_decay": {args.num_epochs - args.warmup_epochs}',
        f'"power": 2',
    ]
    training_command += ["--lr-schedule-params", "{" + ",".join(lr_schedule_params) + "}"]
    # warmup params
    lr_warmup_params = [
        f'"warmup_mode": "shift"',
        f'"warmup_epochs": {args.warmup_epochs}',
    ]
    training_command += ["--lr-warmup-params", "{" + ",".join(lr_warmup_params) + "}"]
    # optimizer params
    optimizer_params = [f'"momentum": {args.opt_momentum}']
    optimizer_params += [f'"weight_decay": {args.lars_weight_decay}'] if args.lars_weight_decay is not None else []
    optimizer_params += [f'"eeta": {args.lars_eeta}', f'"epsilon": 0'] if args.lars_eeta is not None else []
    training_command += ["--optimizer-params", "{" + ",".join(optimizer_params) + "}"]

    training_command = [str(command) for command in training_command]
    print(" ".join(training_command))

    # run training
    p = subprocess.Popen(training_command, stderr=sys.stderr, stdout=sys.stdout)
    p.wait()
