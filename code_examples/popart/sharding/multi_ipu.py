# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import sys

import numpy as np
import popart


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

    input_shape = [opts.batch_size, opts.hidden_size]
    input = builder.addInputTensor(popart.TensorInfo("FLOAT16", input_shape))

    x = input
    if opts.use_zero_values:
        w_data = np.zeros([opts.hidden_size, opts.hidden_size], np.float16)
        b_data = np.zeros([opts.hidden_size], np.float16)
        input_data = np.zeros(input_shape, np.float16)
    else:
        w_data = np.random.normal(0, 1, [opts.hidden_size, opts.hidden_size]).astype(np.float16)
        b_data = np.random.normal(0, 1, [opts.hidden_size]).astype(np.float16)
        input_data = np.random.uniform(-1, 1, input_shape).astype(np.float16)

    for i in range(opts.shards):
        with builder.virtualGraph(i):
            w = builder.addInitializedInputTensor(w_data)
            b = builder.addInitializedInputTensor(b_data)
            x = builder.aiOnnx.gemm([x, w, b])
    output = x
    builder.addOutputTensor(output)

    return [
        builder,
        {input: input_data},
        {output: popart.AnchorReturnType("FINAL")},
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
        label_data = np.random.uniform(0, 2, label_shape).astype(np.int32)

    loss = builder.aiGraphcore.nllloss([probs, label], popart.ReductionType.Sum, debugPrefix="nllLossVal")

    return [
        builder,
        {**data, label: label_data},
        {loss: popart.AnchorReturnType("FINAL")},
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
        popart.SGD(0.01)
    ]


def add_args(parser):
    parser.add_argument("--batch-size", default=1, type=int,
                        help="Number of inputs in a mini-batch")
    parser.add_argument("--hidden-size", default=128, type=int,
                        help="Layer hidden size")
    parser.set_defaults(batches_per_step=1000, steps=5, shards=2)
    return parser


def iteration_report(opts, time):
    return "{:5f} items/sec".format(opts.batch_size * opts.batches_per_step / time)


if __name__ == '__main__':
    sys.path.insert(1, '../../../utils/benchmarks/popart')
    import benchmark

    module = benchmark.Benchmark(
        graph_builder,
        add_args,
        iteration_report
    )

    opts = benchmark.parse_opts(module)

    opts.train = opts.mode == "train"

    # Log Benchmark Message
    print("Popart Multi-IPU {} Synthetic benchmark.\n"
          " Batch size {}.\n"
          " Batches per Step {}.\n"
          " Steps {}.\n"
          " {} IPUs."
          .format(
              {"infer": "Inference", "eval": "Evaluation", "train": "Training"}[opts.mode],
              opts.batch_size,
              opts.batches_per_step if not opts.report else "n/a",
              opts.steps if not opts.report else "n/a",
              opts.shards))
    np.random.seed(42)
    benchmark.run(module, opts)
