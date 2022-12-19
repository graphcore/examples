# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch
import popdist
import horovod.torch as hvd

from tools import str_to_bool


def set_args(args=None):
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model', type=str, default='gpt2', choices=('gpt2-test', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'),
                        help='Select which model to train.')
    parser.add_argument('--max-len', default=128, type=int, required=False,
                        help='The max sequence length.')
    parser.add_argument('--checkpoint-input-dir', default='', type=str, required=False,
                        help='Directory where checkpoints will be load from.')
    parser.add_argument('--checkpoint-output-dir', default=None, type=str, required=False,
                        help='Directory where checkpoints will be saved to.')
    parser.add_argument('--custom-ops', type=str_to_bool, nargs='?', const=True, default=True,
                        help='Enable custom ops.')

    # Dataset
    parser.add_argument('--dataset', type=str, choices=['generated', 'mmap', 'tfrecord', 'pickle'],
                        help="dataset to use for the training.")
    parser.add_argument('--input-files', type=str, required=False,
                        help='Path to the training dataset, or the prefix if using "mmap" dataset.')
    parser.add_argument('--enable-sequence-serialized', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Enable sequence serialization for language model loss.')
    parser.add_argument('--serialized-seq-len', default=128, type=int, required=False,
                        help='Split interval when sequence-serialized is enabled.')
    parser.add_argument('--stride', default=128, type=int, required=False,
                        help='Stride window size to sample dataset.')
    parser.add_argument('--val-num', type=int, default=0,
                        help='Validate dataset length.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Workers for dataloader.')
    parser.add_argument('--async-dataloader', type=str_to_bool, nargs='?', const=True, default=False,
                        help='No Host/IPU I/O, random data created on device.')

    # Train
    parser.add_argument('--epochs', default=1, type=int, required=False,
                        help='Number of epochs to train for.')
    parser.add_argument('--training-steps', default=10000, type=int, required=False,
                        help='Number of steps to train for.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Set the micro-batch-size. This is the single forward-backward path batch-size on one replica.')
    parser.add_argument('--optimizer', default='AdamW', type=str, required=False,
                        help='optimizer to use for the training.')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                        help='Set the weight decay.')
    parser.add_argument('--learning-rate', default=0.00001, type=float, required=False,
                        help='Learning rate value.')
    parser.add_argument('--loss-scaling', default=50000.0, type=float, required=False,
                        help='Loss scaling factor (recommend using powers of 2).')
    parser.add_argument('--auto-loss-scaling', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Enable automatic loss scaling for half precision training.')
    parser.add_argument('--lr-warmup', default=0.1, type=float, required=False,
                        help='fraction of train steps(or --lr-decay-steps) to linearly warmup learning rate over.')
    parser.add_argument('--lr-warmup-steps', default=None, type=int, required=False,
                        help='number of steps to linearly warmup learning rate over.')
    parser.add_argument('--lr-decay-steps', default=None, type=int, required=False,
                        help='number of steps to decay learning rate over, if None defaults to train steps.')
    parser.add_argument('--lr-schedule', default='constant', type=str, choices=('linear', 'constant', 'cosine'), required=False,
                        help='Type of learning rate schedule. --learning-rate will be used as the max value.')
    parser.add_argument('--log-steps', default=1, type=int, required=False,
                        help='Option to print logs after every n training steps.')
    parser.add_argument('--save-per-epochs', default=1, type=int, required=False,
                        help='Option to checkpoint model after every n training epochs.')
    parser.add_argument('--save-per-steps', default=None, type=int, required=False,
                        help='Option to checkpoint model after every n training steps.')
    parser.add_argument('--gradient-accumulation', default=1, type=int, required=False,
                        help='Number of gradients to accumulate before updating the weights.')
    parser.add_argument('--use-wandb', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Enabling logging to Weights and Biases.')

    # Execution
    parser.add_argument('--layers-per-ipu', type=int, nargs='+',
                        help='Number of encoders placed on each IPU. Can be a single number, for an equal number encoder layers per IPU.\
                              Or it can be a list of numbers, specifying number of encoder layers for each individual IPU.')
    parser.add_argument('--device-iterations', default=4, type=int, required=False,
                        help='Number of iterations run on the device before syncing with the host.')
    parser.add_argument('--replication-factor', default=1, type=int, required=False,
                        help='Number of replicas.')
    parser.add_argument('--ipus-per-replica', default=4, type=int, required=False,
                        help='Number of IPUs required by each replica.')
    parser.add_argument('--matmul-proportion', type=float, nargs='+',
                        help='Relative IPU memory proportion size allocated for matmul.')
    parser.add_argument('--recompute-checkpoint-every-layer', type=str_to_bool, nargs='?', const=True, default=False,
                        help='This controls how recomputation is handled in pipelining. '
                             'If True the output of each encoder layer will be stashed keeping the max liveness '
                             'of activations to be at most one layer. '
                             'However, the stash size scales with the number of pipeline stages so this may not always be beneficial. '
                             'The added stash + code could be greater than the reduction in temporary memory.', )
    parser.add_argument('--recompute-checkpoint-layers', type=int, nargs='+', default=None,
                        help='Decoder layers that will be checkpointed.')
    parser.add_argument('--resume-training-from-checkpoint', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Restore both the model checkpoint and optimizer state in order to resume a training run.')
    parser.add_argument('--embedding-serialization-factor', default=1, type=int,
                        help='Matmul serialization factor the embedding layers.')
    parser.add_argument('--remap-logit', type=str_to_bool, nargs='?', const=True, default=False,
                        help='remap logits or not by custom op.')
    parser.add_argument('--optimizer-state-offchip', type=str_to_bool, nargs='?', const=True, default=True,
                        help='Set the tensor storage location for optimizer state to be offchip.')
    parser.add_argument('--replicated-tensor-sharding', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Enable replicated tensor sharding of optimizer state.')
    parser.add_argument('--enable-half-partials', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Enable half partials for matmuls and convolutions globally.')
    parser.add_argument('--executable-cache-dir', default=None, type=str, required=False,
                        help='Directory where Poplar executables are cached. If set, recompilation of identical graphs can be avoided. Required for both saving and loading executables.')
    parser.add_argument('--compile-only', action="store_true",
                        help='Create an offline IPU target that can only be used for offline compilation.')

    args = parser.parse_args(args)
    # Initialise PopDist
    if popdist.isPopdistEnvSet():
        hvd.init()
        args.use_popdist = True
        if popdist.getNumTotalReplicas() != args.replication_factor:
            print(f"The number of replicas is overridden by PopRun. "
                  f"The new value is {popdist.getNumTotalReplicas()}.")
        args.replication_factor = int(popdist.getNumLocalReplicas())
        args.popdist_rank = popdist.getInstanceIndex()
        args.popdist_size = popdist.getNumInstances()

        hvd.broadcast(torch.Tensor([args.seed]), root_rank=0)
    else:
        args.use_popdist = False

    if args.auto_loss_scaling:
        args.loss_scaling = 1.0

    return args
