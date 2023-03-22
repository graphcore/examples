# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse


def parse_params():
    parser = argparse.ArgumentParser(description="Keras U-Net example")
    training_group = parser.add_argument_group("training")
    # Training arguments
    training_group.add_argument("--train", action="store_true", help="Enable IPU training")
    training_group.add_argument(
        "--optimizer", choices=["sgd", "adam"], default="sgd", help="Choose optimizer used in training"
    )
    training_group.add_argument("--eval", action="store_true", help="Enable IPU validation")
    training_group.add_argument("--infer", action="store_true", help="Enable IPU inference")
    training_group.add_argument("--drop-rate", type=float, default=0.5, help="Dropout probability")
    training_group.add_argument("--learning-rate", type=float, default=0.0001, help="Initial learning rate")
    training_group.add_argument(
        "--gradient-accumulation-count",
        type=int,
        default=8,
        help="The number of gradient accumulated per replica before each weight update. When the model is pipelined, it must be at least twice the number of pipeline stages",
    )
    training_group.add_argument(
        "--steps-per-execution", type=int, default=240, help="The number of micro batches processed per execution"
    )
    training_group.add_argument("--micro-batch-size", type=int, default=1, help="Micro batch size")
    training_group.add_argument("--seed", type=int, default=1, help="Set seed for random functions.")
    training_group.add_argument("--num-epochs", type=int, default=2140, help="The number of training epochs.")
    training_group.add_argument("--loss-scale", type=int, default=128, help="Loss scaling factor.")
    training_group.add_argument("--decay-rate", type=float, default=0.9, help="Decay rate of learning rate schedue.")
    training_group.add_argument(
        "--momentum",
        type=float,
        default=0.99,
        help="Momentum to use for stochastic gradient descent, used for IPU training.",
    )
    training_group.add_argument("--eval-freq", type=int, default=4, help="The number of executions per validation.")
    training_group.add_argument("--kfold", type=int, default=1, help="Set the number of folds for cross validation.")
    training_group.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmarking. No checkpoint and no validation in training. Warmup steps in inference.",
    )
    training_group.add_argument(
        "--model-dir", type=str, default="results", help="Directory to place checkpoints and predictions."
    )
    # IPU system arguments
    ipu_group = parser.add_argument_group("ipu")
    ipu_group.add_argument(
        "--nb-ipus-per-replica",
        type=int,
        default=4,
        help="The number of IPUs to use per replica. Only support 4 for training.",
    )
    ipu_group.add_argument(
        "--internal-exchange-target",
        choices=["memory", "balanced", "cycles"],
        default="balanced",
        help="Set the internal exchange target to memory or balanced can help to reduce memory.",
    )
    ipu_group.add_argument(
        "--available-memory-proportion",
        nargs="+",
        type=float,
        default=[0.7, 0.3, 0.45, 0.7],
        help="Set proportion of memory allocated for matrix multiplies, either 1 value for all IPUs or a list of size the same as the number of IPUs",
    )
    ipu_group.add_argument(
        "--pipeline-scheduler",
        choices=["grouped", "interleaved"],
        default="grouped",
        help="Choose the pipeline scheduler type. Grouped results in significantly better throughput at the expense of memory.",
    )
    ipu_group.add_argument("--replicas", type=int, default=1, help="The number of times to replicate the graph")

    # Data arguments
    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--augment",
        action="store_true",
        help="Use data augmentation: random horizontal and vertical flip, random crop resize and random brightness .",
    )
    data_group.add_argument("--dtype", default="float16", type=str, help="Data type, default float16, else float32")
    data_group.add_argument(
        "--partials-type",
        default="half",
        choices=["half", "float"],
        help="Data type used for intermediate calculations. half is for float16, float means float32",
    )
    data_group.add_argument("--host-generated-data", action="store_true", help="Use randomly generated data on host")
    data_group.add_argument("--nb-classes", type=int, default=2, help="The number of output classes")
    data_group.add_argument("--use-prefetch", action="store_true", help="Use dataset prefetch.")
    data_group.add_argument("--data-dir", type=str, default="data", help="Directory containing TIF image data")

    args = parser.parse_args()
    if not (args.train or args.infer):
        parser.error("At least one of --train or --infer must be provided")

    # Only need it for testing on CPU
    if args.nb_ipus_per_replica == 0:
        args.gradient_accumulation_count = 1

    if args.nb_ipus_per_replica != 4 and args.train:
        parser.error("This model has to use 4 IPUs per replica for training.")
    return args
