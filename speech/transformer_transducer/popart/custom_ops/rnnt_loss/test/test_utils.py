# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import math
import sys
import os
import ctypes
from pathlib import Path
import numpy as np


def load_custom_lib(lib):
    # Change cwd to to the parent directory to allow us to resolve the codelet
    os.chdir(str(Path(__file__).resolve().parent.parent.parent.parent))

    so_path = os.path.join(os.path.dirname(__file__), "../../../build/" + lib + ".so")
    if not os.path.isfile(so_path):
        print("Build the custom ops library with `make` before running this script")
        exit(1)
    ctypes.cdll.LoadLibrary(so_path)
    return so_path


def load_custom_rnnt_op():
    return load_custom_lib("rnnt_loss")


def load_custom_sparse_logsoftmax_op():
    return load_custom_lib("sparse_logsoftmax")


def changeLengths(lengths):
    print(lengths.shape)
    max_len = lengths[0]
    changes = np.random.randint(
        0, high=max(max_len / 4, 1), size=lengths.shape[0]
    ).reshape(lengths.shape)
    return (lengths - changes).astype(np.int32)


def generate_data(opts):
    dtype = np.float32 if opts.precision == "FLOAT" else np.float16
    input_shape = [
        opts.batch_size,
        opts.input_size,
        opts.target_size + 1,
        opts.num_classes,
    ]
    inputs = np.random.uniform(
        low=-1 * opts.logits_scale, high=opts.logits_scale, size=input_shape
    ).astype(dtype)
    input_lengths_data = (
        np.array([opts.input_size] * (opts.batch_size))
        .astype(np.int32)
        .reshape(opts.batch_size)
    )

    if opts.variable_input:
        input_lengths_data = changeLengths(input_lengths_data)
        print(input_lengths_data)
    target = (
        np.random.choice(
            np.arange(1, opts.num_classes), opts.batch_size * opts.target_size
        )
        .astype(np.int32)
        .reshape((opts.batch_size, opts.target_size))
    )
    target_lengths_data = (
        np.array([opts.target_size] * (opts.batch_size))
        .astype(np.int32)
        .reshape(opts.batch_size)
    )

    print(
        inputs.shape, input_lengths_data.shape, target.shape, target_lengths_data.shape
    )
    if opts.variable_input:
        target_lengths_data = changeLengths(target_lengths_data)
        print(target_lengths_data)
    return (inputs, target, input_lengths_data, target_lengths_data)


def getTensorError(tA, pA):
    # pA, tA are corresponding tensors from two models
    ss_err = np.sum((np.array(pA) - np.array(tA)) ** 2)
    ss_pA = np.sum(np.array(pA) ** 2)
    ss_tA = np.sum(np.array(tA) ** 2)
    return ss_err / (math.sqrt(ss_pA * ss_tA) + 1.0e-8)


def parse_args(input_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ipu", help="run on available IPU hardware device", action="store_true"
    )
    parser.add_argument(
        "--input-size", default=317, type=int, help="the maximal size of the input."
    )
    parser.add_argument(
        "--target-size", default=116, type=int, help="the maximal size of the targets."
    )
    parser.add_argument(
        "--precision",
        default="FLOAT16",
        type=str,
        choices=("FLOAT", "FLOAT16"),
    )
    parser.add_argument("--partial32", action="store_true", help="use fp32 partials.")
    parser.add_argument(
        "--num-classes", type=int, default=1024, help="Number of classes"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="number of `chunksize` chunks of data to process per session",
    )
    parser.add_argument(
        "--logits-scale",
        type=int,
        default=1,
        help="Absolute maximum for logits scale: logits will be randomly "
        "generated in the range [-logits_scale ... logits_scale].",
    )

    parser.add_argument(
        "--variable-input", help="Use variable size inputs", action="store_true"
    )

    parser.add_argument(
        "--reduction-type",
        type=str,
        choices=("mean", "sum", "none"),
        default="mean",
        help="Type of reduction to be applied to the loss",
    )
    return parser.parse_args(input_list)


def args_from_params(
    input_size=None,
    target_size=None,
    num_classes=None,
    batch_size=None,
    reduction_type=None,
    precision=None,
    variable_input=None,
    partial32=None,
    logits_scale=None,
    ipu=None,
):

    params = []

    if input_size is not None:
        params += ["--input-size", input_size]
    if target_size is not None:
        params += ["--target-size", target_size]
    if num_classes is not None:
        params += ["--num-classes", num_classes]
    if batch_size is not None:
        params += ["--batch-size", batch_size]
    if reduction_type is not None:
        params += ["--reduction-type", reduction_type]
    if precision is not None:
        params += ["--precision", precision]
    if partial32 is not None:
        params += ["--partial32"]
    if variable_input is not None:
        params += ["--variable-input"]
    if logits_scale is not None:
        params += ["--logits-scale", logits_scale]
    if ipu is not None:
        params += ["--ipu"]

    return parse_args([str(p) for p in params])
