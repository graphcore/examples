# Copyright 2020 Graphcore Ltd.
import sys
from pathlib import Path

import numpy as np
import popart
from resnet_builder import PopartBuilderResNet

# Add benchmark module to path
bench_path = Path(Path(__file__).absolute().parent.parent.parent.parent,
                  'utils')
sys.path.append(str(bench_path))
from benchmarks.popart.benchmark import Benchmark, parse_opts, run


class OptimizedResNet(PopartBuilderResNet):
    def __init__(self, opts):
        if not opts.train and opts.norm_type == 'BATCH':
            # For inference, assume normalization on population parameters
            # reduced to a single linear Ax+B transformation. Also assume that the
            # optimization has been applied that folds this transformation into
            # the previous conv + bias layer so there is no normalization op needed.
            opts.norm_type = 'NONE'
        super(OptimizedResNet, self).__init__(opts)


def graph_builder(opts):
    if opts.mode == 'infer':
        builder_fn = infer_builder
    elif opts.mode == 'eval':
        builder_fn = eval_builder
    elif opts.mode == 'train':
        builder_fn = train_builder
    else:
        raise ValueError("Unknown mode '{}'".format(opts.mode))
    defn = builder_fn(opts)
    defn[0] = defn[0].getModelProto()
    return defn


def infer_builder(opts):
    resnet = OptimizedResNet(opts)
    builder = resnet.builder

    input_shape = [opts.batch_size, 4, 224, 224]
    x = builder.addInputTensor(popart.TensorInfo("FLOAT16", input_shape))
    if opts.use_zero_values:
        x_data = np.zeros(input_shape, np.float16)
    else:
        x_data = np.random.normal(0, 1, input_shape).astype(np.float16)
    output = resnet(x)
    builder.addOutputTensor(output)

    return [
        builder,
        {x: x_data},
        {output: popart.AnchorReturnType("ALL")},
        [],
        None
    ]


def eval_builder(opts):
    builder, data, outputs, __, __ = infer_builder(opts)

    probs = builder.aiOnnx.softmax([list(outputs)[0]])

    label_shape = [opts.batch_size]
    label = builder.addInputTensor(popart.TensorInfo("INT32", label_shape))
    if opts.use_zero_values:
        label_data = np.zeros(label_shape, np.int32)
    else:
        label_data = np.random.uniform(0, 10, label_shape).astype(np.int32)

    loss = popart.NllLoss(probs, label, "nllLossVal")

    return [
        builder,
        {**data, label: label_data},
        {loss.output(0): popart.AnchorReturnType("ALL")},
        [loss],
        None
    ]


def train_builder(opts):
    builder, data, outputs, loss, __ = eval_builder(opts)

    return [
        builder,
        data,
        outputs,
        loss,
        popart.ConstSGD(0.01)
    ]


def add_args(parser):
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Set batch size.')
    parser.add_argument('--size', type=int, choices=[8, 14, 20, 32, 44, 56, 110, 18, 34, 50, 101, 152], default=18,
                        help='Size of Resnet graph.')
    parser.add_argument('--norm-type', choices=["BATCH", "GROUP", "NONE"], default="BATCH",
                        help="Choose which normalization to use after each convolution")
    parser.add_argument('--norm-groups', type=int, default=32,
                        help="Sets the number of groups when using the 'GROUP' norm-type")
    parser.add_argument('--shortcut-type', choices=['A', 'B', 'C'],
                        help="ResNet shortcut type. Defaults to definition specified.")
    parser.set_defaults(batches_per_step=1000, steps=5,
                        mode='eval', auto_sharding=True)
    return parser


def iteration_report(opts, time):
    return "{:5f} images/sec.".format(opts.batch_size * opts.batches_per_step / time)


if __name__ == '__main__':
    module = Benchmark(
        graph_builder,
        add_args,
        iteration_report
    )

    opts = parse_opts(module)

    opts.train = opts.mode == "train"

    # Log Benchmark Message
    print(
        "PopART ResNet{} {} Synthetic benchmark.\n"
        " Batch size {}.\n"
        " Batches per Step {}.\n"
        " Steps {}.".format(
            opts.size,
            {"infer": "Inference", "eval": "Evaluation", "train": "Training"}[opts.mode],
            opts.batch_size,
            opts.batches_per_step if not opts.report else "n/a",
            opts.steps if not opts.report else "n/a"))
    np.random.seed(42)
    run(module, opts)
