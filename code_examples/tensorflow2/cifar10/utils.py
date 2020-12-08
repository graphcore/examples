# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import argparse
from tensorflow.python import ipu


def parse_params(enable_multi_ipu=False, enable_pipelining=False):
    parser = argparse.ArgumentParser(description='CIFAR-10 training in Tensorflow 2', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=16, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    if enable_multi_ipu:
        parser.add_argument('--ipus', type=int, default=2, help='number of IPUs')
    if enable_pipelining:
        parser.add_argument('--gradient-accumulation-count', type=int, default=8, help='The number of times each pipeline stage will be executed')
        parser.add_argument('--model_shard_position', type=int, default=7, help='The layer number, where the model is sharded into two parts (range:1..17)')
    opts = parser.parse_args()
    return opts


def create_ipu_config(training_steps, test_steps, num_replicas=1, num_shards=1):
    cfg = ipu.utils.create_ipu_config()
    ipu_options = ipu.utils.auto_select_ipus(cfg, num_ipus=num_replicas*num_shards)

    ipu_run_config = ipu.ipu_run_config.IPURunConfig(
        num_replicas=num_replicas,
        num_shards=num_shards,
        iterations_per_loop=test_steps,
        ipu_options=ipu_options,
    )

    config = ipu.ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config,
        log_step_count_steps=training_steps,
    )
    return config
