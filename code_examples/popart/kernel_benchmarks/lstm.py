# Copyright 2020 Graphcore Ltd.
import sys
from pathlib import Path

import numpy as np
import popart

# Add benchmark module to path
bench_path = Path(Path(__file__).absolute().parent.parent.parent.parent,
                  'utils')
sys.path.append(str(bench_path))
from benchmarks.popart.benchmark import Benchmark, parse_opts, run


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

    timesteps = opts.timesteps
    batch_size = opts.batch_size
    hidden_size = opts.hidden_size
    input_size = hidden_size
    dType_str = "FLOAT16"
    dType = np.float16

    input_shape = [timesteps, batch_size, input_size]
    d1 = popart.TensorInfo(dType_str, input_shape)
    if opts.use_zero_values:
        d2 = np.zeros([1, 4 * hidden_size, input_size], dType)
        d3 = np.zeros([1, 4 * hidden_size, hidden_size], dType)
        input = np.zeros(input_shape, dType)
    else:
        d2 = np.random.normal(
            0, 1, [1, 4 * hidden_size, input_size]).astype(dType)
        d3 = np.random.normal(
            0, 1, [1, 4 * hidden_size, hidden_size]).astype(dType)
        input = np.random.uniform(-1, 1, input_shape).astype(dType)

    i1 = builder.addInputTensor(d1, "input_sequences")
    i2 = builder.addInitializedInputTensor(d2, "gate_weights")
    i3 = builder.addInitializedInputTensor(d3, "recurrence_weights")
    out, Y_h, Y_c = builder.aiOnnx.lstm([i1, i2, i3], 3, clip=None)
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

    probs = builder.aiOnnx.softmax([list(outputs)[0]])
    label_shape = [opts.timesteps, 1, opts.batch_size]
    label = builder.addInputTensor(popart.TensorInfo("INT32", label_shape))

    loss = builder.aiGraphcore.nllloss(
        [probs, label], popart.ReductionType.Sum, debugPrefix="nllLossVal")
    if opts.use_zero_values:
        label_data = np.zeros(label_shape, np.int32)
    else:
        label_data = np.random.uniform(0, 2, label_shape).astype(np.int32)

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
    parser.add_argument('--timesteps', type=int, default=1,
                        help='Set number of recurrent steps.')
    parser.add_argument('--hidden-size', type=int, default=32,
                        help='Set number of units in hidden layer.')
    parser.set_defaults(batches_per_step=1000, steps=5,
                        mode='infer', auto_sharding=True)
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
    print("PopART LSTM {} Synthetic benchmark.\n"
          " Batch size {}.\n"
          " Batches per Step {}.\n"
          " Steps {}.\n"
          " Hidden size {}.\n"
          " Timesteps {}.\n"
          .format(
              {"infer": "Inference", "eval": "Evaluation",
                  "train": "Training"}[opts.mode],
              opts.batch_size,
              opts.batches_per_step if not opts.report else "n/a",
              opts.steps if not opts.report else "n/a",
              opts.hidden_size,
              opts.timesteps))
    np.random.seed(42)
    run(module, opts)
