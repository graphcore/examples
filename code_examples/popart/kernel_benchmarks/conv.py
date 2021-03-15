# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import sys
from pathlib import Path

import numpy as np
import popart

# Add benchmark module to path
bench_path = Path(Path(__file__).absolute().parent.parent.parent.parent,
                  'utils')
sys.path.append(str(bench_path))
from benchmarks.popart.benchmark import Benchmark, parse_opts, run


def kaiming_init(shape, fan_in, a=5.0, b=3.0):
    # shape = [out_channel, in_channel, size, size]
    stddev = np.sqrt(a) / np.sqrt(fan_in)
    bound = np.sqrt(b) * stddev
    return np.random.uniform(-bound, bound, shape).astype(np.float16)


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
    builder = popart.Builder()

    input_width = opts.input_width
    input_height = opts.input_height
    channel_size = opts.channel_size
    batch_size = opts.batch_size
    filter_number = opts.filter_number
    kernel_size = opts.kernel_size
    padding = opts.padding
    stride = opts.stride
    dType_str = "FLOAT16"
    dType = np.float16

    # input shape in NCHW format
    input_shape = [batch_size, channel_size, input_height, input_width]
    d1 = popart.TensorInfo(dType_str, input_shape)
    if opts.use_zero_values:
        d2 = np.zeros([filter_number, channel_size, kernel_size, kernel_size], dType)
        d3 = np.zeros([filter_number], dType)
        input = np.zeros(input_shape, dType)
    else:
        d2 = kaiming_init([filter_number, channel_size, kernel_size, kernel_size],
                          channel_size*input_height*input_width)
        d3 = kaiming_init([filter_number], channel_size*input_height*input_width)
        input = np.random.uniform(-1, 1, input_shape).astype(dType)

    i1 = builder.addInputTensor(d1, "input_tensor")
    i2 = builder.addInitializedInputTensor(d2, "weights")
    i3 = builder.addInitializedInputTensor(d3, "bias")
    out = builder.aiOnnx.conv([i1, i2, i3], strides=[stride, stride],
                              pads = [padding, padding, padding, padding])
    builder.addOutputTensor(out)

    return [
        builder,
        {i1: input},
        {out: popart.AnchorReturnType("ALL")},
        None,
        None
    ]


def eval_builder(opts):
    builder, data, outputs, __, __ = infer_builder(opts)

    probs = builder.aiOnnx.softmax([list(outputs)][0])
    output_height = (opts.input_height + 2*opts.padding - opts.kernel_size)//opts.stride + 1
    output_width = (opts.input_width + 2*opts.padding - opts.kernel_size)//opts.stride + 1
    output_shape = [opts.batch_size, opts.filter_number, output_height, output_width]
    label = builder.addInputTensor(popart.TensorInfo("FLOAT16", output_shape))
    # Sum of square error
    loss = builder.aiOnnx.sub([label, probs])
    loss = builder.aiOnnx.reducesumsquare([loss])

    if opts.use_zero_values:
        label_data = np.zeros(output_shape, np.int32)
    else:
        label_data = np.random.uniform(0, 2, output_shape).astype(np.int32)

    return [
        builder,
        {**data, label: label_data},
        {loss: popart.AnchorReturnType("ALL")},
        loss,
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
    parser.set_defaults(batches_per_step=1000, steps=5,
                        mode='infer', auto_sharding=True)
    parser.add_argument('--input-width', type=int, default=1920,
                        help='Input width size')
    parser.add_argument('--input-height', type=int, default=1080,
                        help='Input height size')
    parser.add_argument('--channel-size', type=int, default=4,
                        help='Channel size')
    parser.add_argument('--filter-number', type=int, default=64,
                        help='Number of filters')
    parser.add_argument('--kernel-size', type=int, default=4,
                        help='Kernel size')
    parser.add_argument('--padding', type=int, default=3,
                        help='Number of padding')
    parser.add_argument('--stride', type=int, default=2,
                        help='Stride for convolution')
    return parser


def iteration_report(opts, time):
    return "{:5f} items/sec.".format(opts.batch_size * opts.batches_per_step / time)


if __name__ == '__main__':
    module = Benchmark(
        graph_builder,
        add_args,
        iteration_report
    )

    opts = parse_opts(module)

    # Log Benchmark Message
    print("PopART Convolutional layer {} Synthetic benchmark.\n"
          " Batch size {}.\n"
          " Batches per Step {}.\n"
          " Steps {}.\n"
          " Input width {}.\n"
          " Input height {}.\n"
          " Channel {}.\n"
          " Padding {}.\n"
          " Stride {}.\n"
          .format(
              {"infer": "Inference", "eval": "Evaluation",
                  "train": "Training"}[opts.mode],
              opts.batch_size,
              opts.batches_per_step if not opts.report else "n/a",
              opts.steps if not opts.report else "n/a",
              opts.input_width,
              opts.input_height,
              opts.channel_size,
              opts.padding,
              opts.stride))
    np.random.seed(42)
    run(module, opts)
